"""
Evaluator for TaikoChartEstimator

Orchestrates evaluation across all metric types and generates reports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    DecompressionMetrics,
    DifficultyMetrics,
    MILHealthMetrics,
    MonotonicityMetrics,
    StarMetrics,
)

# Type hints (avoid runtime import of version-specific classes)
if TYPE_CHECKING:
    from ..data.v1.dataset import TaikoChartDataset
    from ..model.v1.model import TaikoChartEstimator


def get_components(version: str = "v1"):
    """Dynamically load components based on version."""
    if version == "v2":
        from ..data.v2.dataset import TaikoChartDataset, collate_chart_bags
        from ..model.v2.model import ModelConfig, TaikoChartEstimator
    else:
        from ..data.v1.dataset import TaikoChartDataset, collate_chart_bags
        from ..model.v1.model import ModelConfig, TaikoChartEstimator

    return {
        "TaikoChartDataset": TaikoChartDataset,
        "collate_chart_bags": collate_chart_bags,
        "ModelConfig": ModelConfig,
        "TaikoChartEstimator": TaikoChartEstimator,
    }


class Evaluator:
    """
    Comprehensive evaluator for TaikoChartEstimator.

    Runs all metrics and generates detailed reports.
    """

    def __init__(
        self,
        model: TaikoChartEstimator,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device

        # Initialize metric calculators
        self.difficulty_metrics = DifficultyMetrics()
        self.star_metrics = StarMetrics()
        self.monotonicity_metrics = MonotonicityMetrics()
        self.decompression_metrics = DecompressionMetrics()
        self.mil_health_metrics = MILHealthMetrics()

    @torch.no_grad()
    def run_inference(
        self,
        dataloader: DataLoader,
    ) -> dict:
        """
        Run inference on entire dataset and collect predictions.

        Returns:
            Dict with all predictions and metadata
        """
        self.model.eval()

        results = {
            "pred_difficulty_class": [],
            "true_difficulty_class": [],
            "pred_star": [],
            "true_star": [],
            "raw_score": [],
            "song_ids": [],
            "difficulties": [],
            "is_right_censored": [],
            "is_left_censored": [],
            "attention_weights": [],
            "instance_counts": [],
        }

        for batch in tqdm(dataloader, desc="Running inference"):
            instances = batch["instances"].to(self.device)
            instance_masks = batch["instance_masks"].to(self.device)
            instance_counts = batch["instance_counts"].to(self.device)
            difficulty_class = batch["difficulty_class"].to(self.device)

            output = self.model(
                instances,
                instance_masks,
                instance_counts,
                difficulty_hint=difficulty_class,
                return_attention=True,
            )

            # Collect predictions
            results["pred_difficulty_class"].extend(
                output.difficulty_logits.argmax(dim=-1).cpu().numpy()
            )
            results["true_difficulty_class"].extend(batch["difficulty_class"].numpy())
            results["pred_star"].extend(output.raw_star.cpu().numpy())
            results["true_star"].extend(batch["star"].numpy())
            results["raw_score"].extend(output.raw_score.cpu().numpy())
            results["song_ids"].extend(batch["song_ids"])
            results["difficulties"].extend(batch["difficulties"])
            results["is_right_censored"].extend(batch["is_right_censored"].numpy())
            results["is_left_censored"].extend(batch["is_left_censored"].numpy())
            results["instance_counts"].extend(instance_counts.cpu().numpy())

            # Collect attention weights (average across branches if multi-branch)
            if "average_attention" in output.attention_info:
                results["attention_weights"].extend(
                    output.attention_info["average_attention"].cpu().numpy()
                )

        # Convert to numpy arrays
        for key in [
            "pred_difficulty_class",
            "true_difficulty_class",
            "pred_star",
            "true_star",
            "raw_score",
            "is_right_censored",
            "is_left_censored",
            "instance_counts",
        ]:
            results[key] = np.array(results[key])

        # Note: attention_weights remain as a list since each sample can have
        # different numbers of instances (variable-length attention vectors)

        return results

    def compute_all_metrics(self, results: dict) -> dict:
        """
        Compute all metrics from inference results.

        Returns:
            Dict with all metrics organized by category
        """
        all_metrics = {}

        # Difficulty classification metrics
        all_metrics["difficulty"] = self.difficulty_metrics.compute(
            results["pred_difficulty_class"],
            results["true_difficulty_class"],
        )

        # Star regression metrics
        all_metrics["star"] = self.star_metrics.compute(
            results["pred_star"],
            results["true_star"],
            results["true_difficulty_class"],
            results["is_right_censored"],
            results["is_left_censored"],
        )

        # Monotonicity metrics
        all_metrics["monotonicity"] = self.monotonicity_metrics.compute(
            results["raw_score"],
            results["song_ids"],
            results["difficulties"],
        )

        # Decompression metrics
        all_metrics["decompression"] = self.decompression_metrics.compute(
            results["pred_star"],
            results["true_star"],
            results["true_difficulty_class"],
        )

        # MIL health metrics
        if len(results.get("attention_weights", [])) > 0:
            all_metrics["mil_health"] = self.mil_health_metrics.compute(
                results["attention_weights"],
                results["instance_counts"],
            )

        return all_metrics

    def generate_report(
        self,
        metrics: dict,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate a human-readable report from metrics.

        Returns:
            Report as markdown string
        """
        lines = []
        lines.append("# TaikoChartEstimator Evaluation Report")
        lines.append(f"\nGenerated: {datetime.now().isoformat()}\n")

        # Difficulty Classification
        lines.append("## Difficulty Classification")
        lines.append("")
        d_metrics = metrics.get("difficulty", {})
        lines.append(f"- **Accuracy**: {d_metrics.get('accuracy', 0):.4f}")
        lines.append(
            f"- **Balanced Accuracy**: {d_metrics.get('balanced_accuracy', 0):.4f}"
        )
        lines.append(f"- **Macro F1**: {d_metrics.get('macro_f1', 0):.4f}")
        lines.append(
            f"- **±1 Accuracy**: {d_metrics.get('plus_minus_1_accuracy', 0):.4f}"
        )
        lines.append("")

        # Per-class F1
        lines.append("### Per-Class F1")
        for cls in ["easy", "normal", "hard", "oni", "ura"]:
            f1 = d_metrics.get(f"f1_{cls}", 0)
            lines.append(f"- {cls.capitalize()}: {f1:.4f}")
        lines.append("")

        # Star Regression
        lines.append("## Star Rating Prediction")
        lines.append("")
        s_metrics = metrics.get("star", {})
        lines.append("### Overall")
        lines.append(f"- **MAE**: {s_metrics.get('mae', 0):.4f}")
        lines.append(f"- **RMSE**: {s_metrics.get('rmse', 0):.4f}")
        lines.append(f"- **Spearman ρ**: {s_metrics.get('spearman_rho', 0):.4f}")
        lines.append("")

        lines.append("### Uncensored Samples")
        lines.append(f"- **MAE**: {s_metrics.get('mae_uncensored', 0):.4f}")
        lines.append(
            f"- **Spearman ρ**: {s_metrics.get('spearman_rho_uncensored', 0):.4f}"
        )
        lines.append("")

        lines.append("### Censoring Consistency")
        lines.append(
            f"- **Right Censor Violation Rate**: {s_metrics.get('right_censor_violation_rate', 0):.4f}"
        )
        lines.append(
            f"- **Right Censor Mean Shortfall**: {s_metrics.get('right_censor_mean_shortfall', 0):.4f}"
        )
        lines.append(
            f"- **Left Censor Violation Rate**: {s_metrics.get('left_censor_violation_rate', 0):.4f}"
        )
        lines.append("")

        # Monotonicity
        lines.append("## Within-Song Monotonicity")
        lines.append("")
        m_metrics = metrics.get("monotonicity", {})
        lines.append(
            f"- **Violation Rate**: {m_metrics.get('violation_rate', 0):.4f} ({m_metrics.get('n_violations', 0)}/{m_metrics.get('n_pairs', 0)} pairs)"
        )
        lines.append(
            f"- **Mean Violation Margin**: {m_metrics.get('mean_violation_margin', 0):.4f}"
        )
        lines.append(
            f"- **Mean Kendall τ (within-song)**: {m_metrics.get('mean_kendall_tau_within_song', 0):.4f}"
        )
        lines.append("")

        # Decompression
        lines.append("## 10-Star Decompression")
        lines.append("")
        dec_metrics = metrics.get("decompression", {})
        lines.append(
            f"- **Std (10-star predictions)**: {dec_metrics.get('std_10star', 0):.4f}"
        )
        lines.append(
            f"- **Range (10-star predictions)**: {dec_metrics.get('range_10star', 0):.4f}"
        )
        if "p90_p50_10star" in dec_metrics:
            lines.append(f"- **P90 - P50**: {dec_metrics.get('p90_p50_10star', 0):.4f}")
            lines.append(f"- **P99 - P90**: {dec_metrics.get('p99_p90_10star', 0):.4f}")
        lines.append("")

        # MIL Health
        if "mil_health" in metrics:
            lines.append("## MIL Attention Health")
            lines.append("")
            mil_metrics = metrics["mil_health"]
            lines.append(
                f"- **Mean Attention Entropy**: {mil_metrics.get('mean_attention_entropy', 0):.4f}"
            )
            lines.append(
                f"- **Mean Effective Instances**: {mil_metrics.get('mean_effective_instances', 0):.4f}"
            )
            lines.append(
                f"- **Mean Top-5% Mass**: {mil_metrics.get('mean_top5_mass', 0):.4f}"
            )

            if mil_metrics.get("attention_collapse_warning", False):
                lines.append("")
                lines.append(
                    "> ⚠️ **Warning**: Attention collapse detected! "
                    "Model may be relying on too few instances."
                )
            lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.write_text(report)

        return report

    def evaluate(
        self,
        dataloader: DataLoader,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Run full evaluation pipeline.

        Args:
            dataloader: DataLoader for evaluation data
            output_dir: Optional directory to save results

        Returns:
            Dict with all metrics
        """
        # Run inference
        results = self.run_inference(dataloader)

        # Compute metrics
        metrics = self.compute_all_metrics(results)

        # Generate report
        report = self.generate_report(metrics)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics as JSON
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            metrics_serializable = convert_numpy(metrics)
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_serializable, f, indent=2)

            # Save report
            (output_dir / "report.md").write_text(report)

            print(f"Results saved to {output_dir}")

        return metrics


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    version: str = "v1",
) -> TaikoChartEstimator:
    """
    Load model from checkpoint.

    Supports two formats:
    1. Traditional .pt checkpoint file (contains model_state_dict and config)
    2. HuggingFace save_pretrained directory (saved via model.save_pretrained())

    Args:
        checkpoint_path: Path to checkpoint file or pretrained directory
        device: Device to load model to

    Returns:
        Loaded TaikoChartEstimator model
    """
    checkpoint_path = Path(checkpoint_path)
    components = get_components(version)
    ModelConfig = components["ModelConfig"]
    TaikoChartEstimator = components["TaikoChartEstimator"]

    if checkpoint_path.is_dir():
        # HuggingFace pretrained directory format
        model = TaikoChartEstimator.from_pretrained(
            checkpoint_path,
        ).to(device)
    else:
        # Traditional .pt checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = ModelConfig(**checkpoint["config"])
        model = TaikoChartEstimator(config)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TaikoChartEstimator")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="JacobLinCool/taiko-1000-parsed",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
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
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Model/data version (v1 or v2)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using version: {args.version}")

    # Load version-specific components
    components = get_components(args.version)
    TaikoChartDataset = components["TaikoChartDataset"]
    collate_chart_bags = components["collate_chart_bags"]

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(
        Path(args.checkpoint), device, version=args.version
    )

    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = TaikoChartDataset(
        split=args.split,
        dataset_name=args.dataset,
        window_measures=args.window_measures,
        hop_measures=args.hop_measures,
        max_instances_per_chart=args.max_instances,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chart_bags,
        num_workers=args.num_workers,
    )

    print(f"Evaluating on {len(dataset)} samples...")

    # Run evaluation
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(
        dataloader,
        output_dir=Path(args.output_dir),
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Difficulty Macro-F1: {metrics['difficulty']['macro_f1']:.4f}")
    print(f"Star MAE (uncensored): {metrics['star']['mae_uncensored']:.4f}")
    print(f"Star Spearman ρ: {metrics['star']['spearman_rho']:.4f}")
    print(
        f"Monotonicity Violation Rate: {metrics['monotonicity']['violation_rate']:.4f}"
    )
    print(
        f"10-Star Decompression Std: {metrics['decompression'].get('std_10star', 0):.4f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
