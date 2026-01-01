"""
Main TaikoChartEstimator Model

Combines instance encoder, MIL aggregator, and output heads
into a unified model for difficulty estimation.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ...data.v2.tokenizer import DIFFICULTY_ORDER
from .aggregator import GatedMILAggregator, MILAggregator
from .encoder import InstanceEncoder, TCNInstanceEncoder
from .heads import DifficultyClassifier, MonotonicCalibrator, RawScoreHead


@dataclass
class ModelConfig:
    """Configuration for TaikoChartEstimator."""

    # Instance encoder config
    encoder_type: str = "transformer"  # "transformer" or "tcn"
    d_model: int = 256
    n_encoder_layers: int = 4
    n_heads: int = 4
    d_feedforward: int = 512
    encoder_dropout: float = 0.1
    max_seq_len: int = 128
    encoder_pooling: str = "cls"

    # MIL aggregator config
    aggregator_type: str = "multibranch"  # "multibranch" or "gated"
    n_attention_branches: int = 3
    top_k_ratio: float = 0.1
    stochastic_mask_prob: float = 0.3
    aggregator_dropout: float = 0.1

    # Head config
    n_difficulty_classes: int = 5  # easy, normal, hard, oni, ura
    head_hidden_dim: int = 128
    head_dropout: float = 0.1
    calibrator_method: str = "spline"  # "spline" or "mlp"

    # Star ranges per difficulty
    star_ranges: dict = None

    def __post_init__(self):
        if self.star_ranges is None:
            self.star_ranges = {
                0: (1, 5),  # easy
                1: (1, 7),  # normal
                2: (1, 8),  # hard
                3: (1, 10),  # oni
                4: (1, 10),  # ura
            }
        else:
            # Fix JSON serialization issue: keys become strings, values become lists
            # Convert back to int keys and tuple values
            self.star_ranges = {
                int(k): tuple(v) if isinstance(v, list) else v
                for k, v in self.star_ranges.items()
            }


@dataclass
class ModelOutput:
    """Output from TaikoChartEstimator forward pass."""

    raw_score: torch.Tensor  # [batch] unbounded difficulty score
    difficulty_logits: torch.Tensor  # [batch, n_classes] difficulty logits
    raw_star: torch.Tensor  # [batch] star rating (can be < 1 or > 10)
    display_star: torch.Tensor  # [batch] star rating clipped to range
    attention_info: dict  # MIL attention weights and metrics
    instance_embeddings: torch.Tensor  # [batch, n_instances, d_model] for analysis


class TaikoChartEstimator(nn.Module, PyTorchModelHubMixin):
    """
    MIL-based Taiko chart difficulty estimation model.

    Takes a bag of chart instances (beat-aligned windows) and predicts:
    1. Raw difficulty score (unbounded, ℝ)
    2. Difficulty class (easy/normal/hard/oni/ura)
    3. Star rating (per difficulty, can exceed nominal range)

    Architecture:
    - Instance Encoder: Transformer or TCN to encode each window
    - MIL Aggregator: Multi-branch attention pooling
    - Output Heads: Raw score, classifier, monotonic calibrator
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize model.

        Args:
            config: Model configuration (uses defaults if None)
        """
        super().__init__()

        if config is None:
            config = ModelConfig()
        self.config = config

        # Build instance encoder
        if config.encoder_type == "transformer":
            self.instance_encoder = InstanceEncoder(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_encoder_layers,
                d_feedforward=config.d_feedforward,
                dropout=config.encoder_dropout,
                max_seq_len=config.max_seq_len,
                pooling=config.encoder_pooling,
            )
        else:
            self.instance_encoder = TCNInstanceEncoder(
                d_model=config.d_model,
                n_layers=config.n_encoder_layers,
                dropout=config.encoder_dropout,
            )

        # Build MIL aggregator
        if config.aggregator_type == "multibranch":
            self.aggregator = MILAggregator(
                d_instance=config.d_model,
                n_branches=config.n_attention_branches,
                top_k_ratio=config.top_k_ratio,
                stochastic_mask_prob=config.stochastic_mask_prob,
                dropout=config.aggregator_dropout,
            )
        else:
            self.aggregator = GatedMILAggregator(
                d_instance=config.d_model,
                dropout=config.aggregator_dropout,
            )

        # Output heads
        bag_dim = self.aggregator.output_dim

        self.raw_score_head = RawScoreHead(
            d_input=bag_dim,
            d_hidden=config.head_hidden_dim,
            dropout=config.head_dropout,
        )

        self.difficulty_classifier = DifficultyClassifier(
            d_input=bag_dim,
            n_classes=config.n_difficulty_classes,
            d_hidden=config.head_hidden_dim,
            dropout=config.head_dropout,
        )

        self.calibrator = MonotonicCalibrator(
            method=config.calibrator_method,
            n_difficulties=config.n_difficulty_classes,
            star_ranges=config.star_ranges,
        )

    def encode_instances(
        self,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all instances in a batch.

        Args:
            instances: [batch, n_instances, seq_len, 6] token sequences
            instance_masks: [batch, n_instances, seq_len] attention masks

        Returns:
            instance_embeddings: [batch, n_instances, d_model]
        """
        batch_size, n_instances, seq_len, n_features = instances.shape

        # Flatten batch and instances
        flat_instances = instances.view(batch_size * n_instances, seq_len, n_features)
        flat_masks = instance_masks.view(batch_size * n_instances, seq_len)

        # Encode
        flat_embeddings = self.instance_encoder(flat_instances, flat_masks)

        # Reshape back
        instance_embeddings = flat_embeddings.view(batch_size, n_instances, -1)

        return instance_embeddings

    def forward(
        self,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
        instance_counts: Optional[torch.Tensor] = None,
        difficulty_hint: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> ModelOutput:
        """
        Forward pass through the model.

        Args:
            instances: [batch, n_instances, seq_len, 6] token sequences
            instance_masks: [batch, n_instances, seq_len] token masks
            instance_counts: [batch] number of valid instances per sample
            difficulty_hint: [batch] difficulty class for calibration (uses predicted if None)
            return_attention: Whether to return attention weights

        Returns:
            ModelOutput with all predictions
        """
        batch_size, n_instances, seq_len, _ = instances.shape

        # Create instance-level mask from counts
        if instance_counts is not None:
            bag_mask = torch.arange(n_instances, device=instances.device).unsqueeze(0)
            bag_mask = (bag_mask < instance_counts.unsqueeze(1)).float()
        else:
            # Infer from instance masks (if any token is valid, instance is valid)
            bag_mask = (instance_masks.sum(dim=-1) > 0).float()

        # Encode instances
        instance_embeddings = self.encode_instances(instances, instance_masks)

        # Aggregate to bag embedding
        bag_embedding, attention_info = self.aggregator(
            instance_embeddings,
            bag_mask,
            return_attention=return_attention,
        )

        # Raw score prediction (unbounded)
        raw_score = self.raw_score_head(bag_embedding)

        # Difficulty classification
        difficulty_logits = self.difficulty_classifier(bag_embedding)

        # Determine difficulty for calibration
        if difficulty_hint is not None:
            calibration_diff = difficulty_hint
        else:
            calibration_diff = difficulty_logits.argmax(dim=-1)

        # Calibrate to star rating
        raw_star = self.calibrator(raw_score, calibration_diff)
        display_star = self.calibrator.clip_to_display(raw_star, calibration_diff)

        return ModelOutput(
            raw_score=raw_score,
            difficulty_logits=difficulty_logits,
            raw_star=raw_star,
            display_star=display_star,
            attention_info=attention_info,
            instance_embeddings=instance_embeddings,
        )

    def get_instance_scores(
        self,
        instance_embeddings: torch.Tensor,
        difficulty_class_id: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate difficulty score for each individual instance.

        This acts as a "probe": we ask the model "if the whole song consisted
        only of this specific instance, what would the difficulty be?"

        Args:
            instance_embeddings: [batch, n_instances, d_model]
            difficulty_class_id: [batch] Optional difficulty class for calibration

        Returns:
            raw_scores: [batch, n_instances] Unbounded raw scores
            star_ratings: [batch, n_instances] Calibrated star ratings
        """
        batch_size, n_instances, _ = instance_embeddings.shape

        # We need to pass each instance through the aggregator's fusion layer.
        # The aggregator usually combines Mean, Top-K, and Branch outputs.
        # For a single-instance bag:
        # - Mean pooling = the instance itself
        # - Top-K pooling = the instance itself
        # - Branch pooling = the instance itself (weighted by 1.0)

        # So we can construct the fused input directly.
        # Concatenation order in MILAggregator: [mean, topk, branch_1, ..., branch_n]

        # [batch, n_instances, d_instance]
        feat = instance_embeddings

        # Construct the concatenated feature vector for a "single-instance bag"
        # We repeat the feature for: Mean (1) + TopK (1) + Branches (n_branches)
        # Total repeats = 2 + n_branches
        if hasattr(self.aggregator, "n_branches"):
            n_repeats = 2 + self.aggregator.n_branches
            # fused_input: [batch, n_instances, d_instance * n_repeats]
            fused_input = feat.repeat(1, 1, n_repeats)

            # Pass through fusion layer
            # fusion expects [..., input_dim], so we can pass (batch * n_inst)
            flat_input = fused_input.view(-1, fused_input.size(-1))
            bag_embedding = self.aggregator.fusion(
                flat_input
            )  # [batch * n_inst, output_dim]
        elif isinstance(
            self.aggregator, type(self).GatedMILAggregator
        ):  # Check if Gated
            # Gated aggregator output projection
            # Gated aggregation of 1 instance is just the instance projected
            flat_feat = feat.view(-1, feat.size(-1))
            bag_embedding = self.aggregator.output_proj(flat_feat)
        else:
            # Fallback for generic/unknown aggregator
            # Assume we can just run the aggregator on size-1 bags?
            # But that's slow. Let's try to simulate if simple enough.
            # For now, raise or return zeros if unknown.
            return torch.zeros_like(feat[..., 0]), torch.zeros_like(feat[..., 0])

        # Raw score head
        raw_score = self.raw_score_head(bag_embedding)  # [batch * n_inst, 1]
        raw_score = raw_score.view(batch_size, n_instances)

        # Calibration
        # If no difficulty provided, predict it from the single instance
        if difficulty_class_id is None:
            logits = self.difficulty_classifier(bag_embedding)
            diff_ids = logits.argmax(dim=-1)  # [batch * n_inst]
        else:
            # Expand provided difficulty to per-instance
            diff_ids = difficulty_class_id.unsqueeze(1).repeat(1, n_instances).view(-1)

        # Calibrate
        flat_raw = raw_score.view(-1)
        stars = self.calibrator(flat_raw, diff_ids)  # [batch * n_inst]
        stars = stars.view(batch_size, n_instances)

        return raw_score, stars

    def predict(
        self,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
        instance_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Convenience method for inference.

        Returns dict with human-readable outputs:
        - difficulty_class: Predicted difficulty name
        - raw_score: Unbounded difficulty score
        - raw_star: Star rating (may exceed range)
        - display_star: Star rating for display (clipped)
        """
        output = self.forward(
            instances,
            instance_masks,
            instance_counts,
            difficulty_hint=None,
            return_attention=False,
        )

        difficulty_names = ["easy", "normal", "hard", "oni", "ura"]
        predicted_class = output.difficulty_logits.argmax(dim=-1)

        return {
            "difficulty_class": [difficulty_names[c] for c in predicted_class.tolist()],
            "difficulty_class_id": predicted_class,
            "raw_score": output.raw_score,
            "raw_star": output.raw_star,
            "display_star": output.display_star,
        }

    def get_ranking_pairs_from_batch(
        self,
        raw_scores: torch.Tensor,
        song_ids: list[str],
        difficulties: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract within-song ranking pairs from a batch.

        Args:
            raw_scores: [batch] raw difficulty scores
            song_ids: List of song IDs
            difficulties: List of difficulty names

        Returns:
            (s_easier, s_harder) tensors for ranking loss
        """

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

            # Create pairs
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

        return (
            torch.stack(easier_scores),
            torch.stack(harder_scores),
        )


def create_model(
    d_model: int = 256,
    n_layers: int = 4,
    encoder_type: str = "transformer",
    **kwargs,
) -> TaikoChartEstimator:
    """
    Factory function to create model with common configurations.

    Args:
        d_model: Model dimension
        n_layers: Number of encoder layers
        encoder_type: "transformer" or "tcn"
        **kwargs: Additional config overrides

    Returns:
        Configured TaikoChartEstimator
    """
    config = ModelConfig(
        encoder_type=encoder_type,
        d_model=d_model,
        n_encoder_layers=n_layers,
        **kwargs,
    )
    return TaikoChartEstimator(config)
