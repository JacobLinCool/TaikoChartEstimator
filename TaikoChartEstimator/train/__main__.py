"""
Training Script for TaikoChartEstimator

Main entry point for training the MIL-based difficulty estimation model.
Supports:
- Multi-task learning (classification + regression + ranking)
- Curriculum learning for loss weights
- TensorBoard logging
- Multi-objective checkpoint selection
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data import TaikoChartDataset, WithinSongPairSampler, collate_chart_bags
from ..data.tokenizer import DIFFICULTY_ORDER
from ..model import CurriculumScheduler, ModelConfig, TaikoChartEstimator, TotalLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train TaikoChartEstimator")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="JacobLinCool/taiko-1000-parsed",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Cache directory for dataset"
    )
    parser.add_argument(
        "--include-audio", action="store_true", help="Include audio features (slower)"
    )

    # Model arguments
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--n-layers", type=int, default=4, help="Number of encoder layers"
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="transformer",
        choices=["transformer", "tcn"],
        help="Instance encoder type",
    )
    parser.add_argument(
        "--n-branches", type=int, default=3, help="Number of attention branches in MIL"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping norm"
    )

    # Loss weights
    parser.add_argument(
        "--lambda-cls", type=float, default=1.0, help="Classification loss weight"
    )
    parser.add_argument(
        "--lambda-star", type=float, default=1.0, help="Star regression loss weight"
    )
    parser.add_argument(
        "--lambda-rank", type=float, default=1.0, help="Ranking loss weight"
    )
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Use curriculum learning for loss weights",
    )

    # Checkpointing and logging
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default="runs", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--eval-every", type=int, default=1, help="Evaluate every N epochs"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--overfit-batch",
        action="store_true",
        help="Overfit on a single batch (for debugging)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=16, help="Number of data loader workers"
    )
    parser.add_argument(
        "--window-measures",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Window sizes in measures (default: 2 4)",
    )
    parser.add_argument(
        "--hop-measures",
        type=int,
        default=2,
        help="Window hop size in measures (default: 2)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=64,
        help="Maximum instances (windows) per chart (default: 64)",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_class_weights(
    dataset: TaikoChartDataset, merge_ura_oni: bool = True
) -> torch.Tensor:
    """Compute class weights based on class frequencies.

    Args:
        dataset: The training dataset
        merge_ura_oni: If True, treat ura and oni as the same class (4 classes total)

    Returns:
        Class weights tensor (4 or 5 weights depending on merge_ura_oni)
    """
    n_classes = 4 if merge_ura_oni else 5
    class_counts = [0] * n_classes

    for song_idx, diff in dataset.chart_index:
        diff_id = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4}.get(diff, 0)
        # Merge ura into oni if enabled
        if merge_ura_oni and diff_id == 4:
            diff_id = 3
        class_counts[diff_id] += 1

    total = sum(class_counts)
    weights = [
        total / (n_classes * count) if count > 0 else 1.0 for count in class_counts
    ]

    return torch.tensor(weights, dtype=torch.float32)


def extract_ranking_pairs(
    batch: dict, raw_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract within-song ranking pairs from batch."""
    song_ids = batch["song_ids"]
    difficulties = batch["difficulties"]

    # Group by song
    song_to_indices: dict[str, list[int]] = {}
    for i, song_id in enumerate(song_ids):
        if song_id not in song_to_indices:
            song_to_indices[song_id] = []
        song_to_indices[song_id].append(i)

    easier_scores = []
    harder_scores = []

    for song_id, indices in song_to_indices.items():
        if len(indices) < 2:
            continue

        # Sort by difficulty
        sorted_indices = sorted(
            indices, key=lambda i: DIFFICULTY_ORDER.get(difficulties[i], 0)
        )

        # Create adjacent pairs
        for i in range(len(sorted_indices) - 1):
            easier_idx = sorted_indices[i]
            harder_idx = sorted_indices[i + 1]

            easier_scores.append(raw_scores[easier_idx])
            harder_scores.append(raw_scores[harder_idx])

    if not easier_scores:
        return (
            torch.tensor([], device=raw_scores.device),
            torch.tensor([], device=raw_scores.device),
        )

    return torch.stack(easier_scores), torch.stack(harder_scores)


def train_epoch(
    model: TaikoChartEstimator,
    dataloader: DataLoader,
    criterion: TotalLoss,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    curriculum: Optional[CurriculumScheduler] = None,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_star_loss = 0.0
    total_rank_loss = 0.0
    n_batches = 0
    n_ranking_pairs = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = epoch * len(dataloader) + batch_idx

        # Update curriculum weights
        if curriculum is not None:
            weights = curriculum.get_weights(global_step)
            criterion.set_weights(**weights)

        # Move batch to device
        instances = batch["instances"].to(device)
        instance_masks = batch["instance_masks"].to(device)
        instance_counts = batch["instance_counts"].to(device)
        difficulty_class = batch["difficulty_class"].to(device)
        star = batch["star"].to(device)
        is_right_censored = batch["is_right_censored"].to(device)
        is_left_censored = batch["is_left_censored"].to(device)

        # Forward pass
        output = model(
            instances,
            instance_masks,
            instance_counts,
            difficulty_hint=difficulty_class,  # Use ground truth for training
        )

        # Extract ranking pairs
        s_easier, s_harder = extract_ranking_pairs(batch, output.raw_score)
        ranking_pairs = (s_easier, s_harder) if s_easier.numel() > 0 else None
        n_ranking_pairs += s_easier.numel()

        # Compute losses
        losses = criterion(
            difficulty_logits=output.difficulty_logits,
            pred_star=output.raw_star,
            target_difficulty=difficulty_class,
            target_star=star,
            is_right_censored=is_right_censored,
            is_left_censored=is_left_censored,
            ranking_pairs=ranking_pairs,
        )

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track losses
        total_loss += losses["total"].item()
        total_cls_loss += losses["cls"].item()
        total_star_loss += losses["star"].item()
        total_rank_loss += losses["rank"].item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{losses['total'].item():.4f}",
                "cls": f"{losses['cls'].item():.4f}",
                "star": f"{losses['star'].item():.4f}",
                "rank": f"{losses['rank'].item():.4f}",
            }
        )

        # Log to TensorBoard
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
            writer.add_scalar("train/loss_cls", losses["cls"].item(), global_step)
            writer.add_scalar("train/loss_star", losses["star"].item(), global_step)
            writer.add_scalar("train/loss_rank", losses["rank"].item(), global_step)

            # Log attention health metrics
            if "entropy" in output.attention_info:
                writer.add_scalar(
                    "train/attention_entropy",
                    output.attention_info["entropy"].mean().item(),
                    global_step,
                )
            if "effective_n" in output.attention_info:
                writer.add_scalar(
                    "train/effective_instances",
                    output.attention_info["effective_n"].mean().item(),
                    global_step,
                )
            if "top5_mass" in output.attention_info:
                writer.add_scalar(
                    "train/top5_attention_mass",
                    output.attention_info["top5_mass"].mean().item(),
                    global_step,
                )

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "star_loss": total_star_loss / n_batches,
        "rank_loss": total_rank_loss / n_batches,
        "n_ranking_pairs": n_ranking_pairs,
    }


@torch.no_grad()
def evaluate(
    model: TaikoChartEstimator,
    dataloader: DataLoader,
    criterion: TotalLoss,
    device: torch.device,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()

    all_pred_class = []
    all_true_class = []
    all_pred_star = []
    all_true_star = []
    all_raw_scores = []
    all_difficulties = []
    all_song_ids = []
    all_is_right_censored = []

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        instances = batch["instances"].to(device)
        instance_masks = batch["instance_masks"].to(device)
        instance_counts = batch["instance_counts"].to(device)
        difficulty_class = batch["difficulty_class"].to(device)
        star = batch["star"].to(device)
        is_right_censored = batch["is_right_censored"].to(device)
        is_left_censored = batch["is_left_censored"].to(device)

        output = model(
            instances,
            instance_masks,
            instance_counts,
            difficulty_hint=difficulty_class,
        )

        # Compute loss
        losses = criterion(
            difficulty_logits=output.difficulty_logits,
            pred_star=output.raw_star,
            target_difficulty=difficulty_class,
            target_star=star,
            is_right_censored=is_right_censored,
            is_left_censored=is_left_censored,
        )

        total_loss += losses["total"].item()
        n_batches += 1

        # Collect predictions
        all_pred_class.extend(output.difficulty_logits.argmax(dim=-1).cpu().tolist())
        all_true_class.extend(difficulty_class.cpu().tolist())
        all_pred_star.extend(output.raw_star.cpu().tolist())
        all_true_star.extend(star.cpu().tolist())
        all_raw_scores.extend(output.raw_score.cpu().tolist())
        all_difficulties.extend(batch["difficulties"])
        all_song_ids.extend(batch["song_ids"])
        all_is_right_censored.extend(is_right_censored.cpu().tolist())

    # Compute metrics
    all_pred_class = np.array(all_pred_class)
    all_true_class = np.array(all_true_class)
    all_pred_star = np.array(all_pred_star)
    all_true_star = np.array(all_true_star)
    all_raw_scores = np.array(all_raw_scores)
    all_is_right_censored = np.array(all_is_right_censored)

    # Merge ura (4) and oni (3) for classification metrics
    # They are essentially the same difficulty level
    all_pred_class_merged = all_pred_class.copy()
    all_true_class_merged = all_true_class.copy()
    all_pred_class_merged[all_pred_class_merged == 4] = 3  # Map ura -> oni
    all_true_class_merged[all_true_class_merged == 4] = 3  # Map ura -> oni

    # Classification metrics (using merged classes)
    macro_f1 = f1_score(all_true_class_merged, all_pred_class_merged, average="macro")
    balanced_acc = balanced_accuracy_score(all_true_class_merged, all_pred_class_merged)
    plus_minus_1_acc = (
        np.abs(all_pred_class_merged - all_true_class_merged) <= 1
    ).mean()

    # Per-difficulty classification metrics (precision, recall, F1)
    diff_names_cls = ["easy", "normal", "hard", "oni_ura"]
    per_diff_cls_metrics = {}

    per_class_f1 = f1_score(
        all_true_class_merged, all_pred_class_merged, average=None, labels=[0, 1, 2, 3]
    )
    per_class_precision = precision_score(
        all_true_class_merged,
        all_pred_class_merged,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )
    per_class_recall = recall_score(
        all_true_class_merged,
        all_pred_class_merged,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    for i, name in enumerate(diff_names_cls):
        if i < len(per_class_f1):
            per_diff_cls_metrics[f"f1_{name}"] = per_class_f1[i]
            per_diff_cls_metrics[f"precision_{name}"] = per_class_precision[i]
            per_diff_cls_metrics[f"recall_{name}"] = per_class_recall[i]

    # Star regression metrics (on uncensored samples)
    uncensored_mask = ~all_is_right_censored
    if uncensored_mask.sum() > 0:
        mae_uncensored = np.abs(
            all_pred_star[uncensored_mask] - all_true_star[uncensored_mask]
        ).mean()
        spearman_rho, _ = spearmanr(all_pred_star, all_true_star)
    else:
        mae_uncensored = 0.0
        spearman_rho = 0.0

    # Per-difficulty Star MAE & RMSE (using merged oni/ura as same class)
    diff_names_merged = ["easy", "normal", "hard", "oni_ura"]
    per_diff_star_metrics = {}

    for diff_idx, diff_name in enumerate(diff_names_merged):
        if diff_idx == 3:
            # oni_ura: merge classes 3 and 4
            mask = (all_true_class == 3) | (all_true_class == 4)
        else:
            mask = all_true_class == diff_idx

        if mask.sum() > 0:
            diff_pred = all_pred_star[mask]
            diff_true = all_true_star[mask]
            diff_errors = diff_pred - diff_true

            per_diff_star_metrics[f"mae_star_{diff_name}"] = np.abs(diff_errors).mean()
            per_diff_star_metrics[f"rmse_star_{diff_name}"] = np.sqrt(
                (diff_errors**2).mean()
            )
        else:
            per_diff_star_metrics[f"mae_star_{diff_name}"] = 0.0
            per_diff_star_metrics[f"rmse_star_{diff_name}"] = 0.0

    # Monotonicity metrics
    song_groups: dict[str, list] = {}
    for i, song_id in enumerate(all_song_ids):
        if song_id not in song_groups:
            song_groups[song_id] = []
        song_groups[song_id].append(
            {
                "difficulty": all_difficulties[i],
                "raw_score": all_raw_scores[i],
            }
        )

    n_violations = 0
    n_pairs = 0

    for song_id, charts in song_groups.items():
        if len(charts) < 2:
            continue

        sorted_charts = sorted(
            charts, key=lambda c: DIFFICULTY_ORDER.get(c["difficulty"], 0)
        )

        for i in range(len(sorted_charts) - 1):
            n_pairs += 1
            if sorted_charts[i]["raw_score"] >= sorted_charts[i + 1]["raw_score"]:
                n_violations += 1

    violation_rate = n_violations / n_pairs if n_pairs > 0 else 0.0

    # Decompression metrics (for 10-star samples)
    max_star_mask = all_true_star >= 10.0
    if max_star_mask.sum() > 1:
        pred_10star = all_pred_star[max_star_mask]
        decompression_std = pred_10star.std()
        p90_p50 = np.percentile(pred_10star, 90) - np.percentile(pred_10star, 50)
    else:
        decompression_std = 0.0
        p90_p50 = 0.0

    result = {
        "loss": total_loss / n_batches,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "plus_minus_1_accuracy": plus_minus_1_acc,
        "mae_uncensored": mae_uncensored,
        "spearman_rho": spearman_rho,
        "monotonicity_violation_rate": violation_rate,
        "decompression_std": decompression_std,
        "decompression_p90_p50": p90_p50,
    }
    # Add per-difficulty classification metrics
    result.update(per_diff_cls_metrics)
    # Add per-difficulty star metrics
    result.update(per_diff_star_metrics)

    return result


def save_checkpoint(
    model: TaikoChartEstimator,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    output_dir: Path,
    name: str = "checkpoint",
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": model.config.__dict__,
    }

    pretrained_path = output_dir / "pretrained" / name
    model.save_pretrained(pretrained_path)

    path = output_dir / f"{name}_epoch{epoch}.pt"
    torch.save(checkpoint, path)

    # Also save as latest
    latest_path = output_dir / f"{name}_latest.pt"
    torch.save(checkpoint, latest_path)

    return path


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(args.tensorboard_dir) / timestamp
    writer = SummaryWriter(tensorboard_dir)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"TensorBoard directory: {tensorboard_dir}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = TaikoChartDataset(
        split="train",
        dataset_name=args.dataset,
        include_audio=args.include_audio,
        cache_dir=args.cache_dir,
        window_measures=args.window_measures,
        hop_measures=args.hop_measures,
        max_instances_per_chart=args.max_instances,
    )

    val_dataset = TaikoChartDataset(
        split="test",
        dataset_name=args.dataset,
        include_audio=args.include_audio,
        cache_dir=args.cache_dir,
        window_measures=args.window_measures,
        hop_measures=args.hop_measures,
        max_instances_per_chart=args.max_instances,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create data loaders
    if args.overfit_batch:
        # Take a small subset for debugging
        train_dataset = Subset(train_dataset, list(range(min(32, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(8, len(val_dataset)))))

    train_sampler = WithinSongPairSampler(
        train_dataset
        if not isinstance(train_dataset, torch.utils.data.Subset)
        else train_dataset.dataset,
        min_batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler if not args.overfit_batch else None,
        batch_size=args.batch_size if args.overfit_batch else 1,
        shuffle=args.overfit_batch,
        collate_fn=collate_chart_bags,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chart_bags,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    config = ModelConfig(
        encoder_type=args.encoder_type,
        d_model=args.d_model,
        n_encoder_layers=args.n_layers,
        n_attention_branches=args.n_branches,
    )
    model = TaikoChartEstimator(config)
    model = model.to(args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Create loss function
    class_weights = compute_class_weights(
        train_dataset
        if not isinstance(train_dataset, torch.utils.data.Subset)
        else train_dataset.dataset
    ).to(args.device)

    criterion = TotalLoss(
        lambda_cls=args.lambda_cls,
        lambda_star=args.lambda_star,
        lambda_rank=args.lambda_rank,
        class_weights=class_weights,
    )

    # Curriculum scheduler
    curriculum = None
    if args.use_curriculum:
        total_steps = args.epochs * len(train_loader)
        curriculum = CurriculumScheduler(total_steps)

    # Composite score function for model selection
    def compute_composite_score(metrics: dict) -> float:
        """
        Compute weighted composite score for model selection.

        Weights prioritize Spearman (star ranking) as the core objective.
        - Spearman ρ: 55% (star prediction ranking accuracy)
        - Macro-F1: 25% (difficulty classification)
        - Violation Rate: 20% (monotonicity constraint)
        """
        # Clamp to reasonable ranges observed in training
        f1 = max(0.70, min(0.90, metrics["macro_f1"]))
        spearman = max(0.80, min(0.98, metrics["spearman_rho"]))
        violation = max(0.0, min(0.10, metrics["monotonicity_violation_rate"]))

        # Normalize to 0-1
        f1_norm = (f1 - 0.70) / 0.20
        spearman_norm = (spearman - 0.80) / 0.18
        violation_norm = 1.0 - violation / 0.10  # Lower is better

        return 0.6 * spearman_norm + 0.25 * f1_norm + 0.15 * violation_norm

    # Training loop
    print("Starting training...")
    best_metrics = {
        "macro_f1": 0.0,
        "spearman_rho": 0.0,
        "monotonicity_violation_rate": 1.0,
    }
    best_composite_score = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device(args.device),
            epoch=epoch,
            writer=writer,
            curriculum=curriculum,
            grad_clip=args.grad_clip,
        )

        print(
            f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
            f"Cls: {train_metrics['cls_loss']:.4f}, Star: {train_metrics['star_loss']:.4f}, "
            f"Rank: {train_metrics['rank_loss']:.4f} ({train_metrics['n_ranking_pairs']} pairs)"
        )

        # Log training metrics
        writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
        writer.add_scalar("epoch/learning_rate", scheduler.get_last_lr()[0], epoch)

        # Evaluate
        if epoch % args.eval_every == 0:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=torch.device(args.device),
            )

            # Compute composite score
            composite_score = compute_composite_score(val_metrics)

            print(
                f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                f"Macro-F1: {val_metrics['macro_f1']:.4f}, "
                f"Spearman: {val_metrics['spearman_rho']:.4f}, "
                f"Violation Rate: {val_metrics['monotonicity_violation_rate']:.4f}, "
                f"Decomp Std: {val_metrics['decompression_std']:.4f}, "
                f"Composite: {composite_score:.4f}"
            )

            # Log validation metrics
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("val/composite_score", composite_score, epoch)

            # Save best model based on composite score
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_metrics = val_metrics
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, output_dir, "best"
                )
                print(f"  -> New best model saved! (Composite: {composite_score:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics, output_dir, "checkpoint"
            )

    # Save final model
    save_checkpoint(model, optimizer, args.epochs, best_metrics, output_dir, "final")

    print(f"\nTraining complete!")
    print(f"Best Composite Score: {best_composite_score:.4f}")
    print(f"  - Macro-F1: {best_metrics['macro_f1']:.4f}")
    print(f"  - Spearman: {best_metrics['spearman_rho']:.4f}")
    print(f"  - Violation Rate: {best_metrics['monotonicity_violation_rate']:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    writer.close()


if __name__ == "__main__":
    main()
