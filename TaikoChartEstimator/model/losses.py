"""
Loss Functions for Taiko Chart Estimation

Implements:
- Within-song ranking loss (monotonicity constraint)
- Censored regression loss (handles star boundary labels)
- Multi-task loss combiner with curriculum scheduling
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import STAR_RANGES_BY_ID as STAR_RANGES


class WithinSongRankingLoss(nn.Module):
    """
    Ranking loss for enforcing within-song monotonicity.

    For charts from the same song, harder difficulties must have
    higher raw scores: s_harder > s_easier.

    Uses hinge loss: L = max(0, margin - (s_harder - s_easier))
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Minimum required difference between difficulty levels
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        s_easier: torch.Tensor,
        s_harder: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ranking loss for pairs.

        Args:
            s_easier: [n_pairs] scores for easier charts
            s_harder: [n_pairs] scores for harder charts

        Returns:
            Scalar loss value
        """
        if s_easier.numel() == 0:
            return torch.tensor(0.0, device=s_easier.device)

        # Hinge loss
        violations = F.relu(self.margin - (s_harder - s_easier))

        return violations.mean()

    def compute_violation_rate(
        self,
        s_easier: torch.Tensor,
        s_harder: torch.Tensor,
    ) -> float:
        """Compute fraction of pairs that violate monotonicity."""
        if s_easier.numel() == 0:
            return 0.0

        violations = (s_easier >= s_harder).float()
        return violations.mean().item()


class CensoredRegressionLoss(nn.Module):
    """
    Censored regression loss for star ratings.

    Handles the fact that boundary labels (1, 10) are censored:
    - label == max_star: true value is >= max_star (right-censored)
    - label == min_star: true value is <= min_star (left-censored)

    For censored samples, we only penalize predictions that
    violate the bound, not predictions that exceed it.
    """

    def __init__(
        self,
        uncensored_loss: str = "huber",  # "huber", "mse", "mae"
        huber_delta: float = 0.5,
        star_ranges: Optional[dict] = None,
    ):
        """
        Args:
            uncensored_loss: Loss type for uncensored samples
            huber_delta: Delta for Huber loss
            star_ranges: Dict mapping difficulty index to (min, max) range
        """
        super().__init__()

        self.uncensored_loss = uncensored_loss
        self.huber_delta = huber_delta
        self.star_ranges = star_ranges if star_ranges is not None else STAR_RANGES

    def _uncensored_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for uncensored samples."""
        if self.uncensored_loss == "huber":
            return F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")
        elif self.uncensored_loss == "mse":
            return F.mse_loss(pred, target, reduction="none")
        elif self.uncensored_loss == "mae":
            return F.l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.uncensored_loss}")

    def forward(
        self,
        pred_star: torch.Tensor,
        target_star: torch.Tensor,
        difficulty: torch.Tensor,
        is_right_censored: Optional[torch.Tensor] = None,
        is_left_censored: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute censored regression loss.

        Args:
            pred_star: [batch] predicted star ratings
            target_star: [batch] target star labels
            difficulty: [batch] difficulty class indices
            is_right_censored: [batch] bool, True if label is at max (right-censored)
            is_left_censored: [batch] bool, True if label is at min (left-censored)

        Returns:
            Scalar loss value
        """
        batch_size = pred_star.size(0)

        # Auto-detect censoring if not provided
        if is_right_censored is None or is_left_censored is None:
            is_right_censored = torch.zeros(
                batch_size, dtype=torch.bool, device=pred_star.device
            )
            is_left_censored = torch.zeros(
                batch_size, dtype=torch.bool, device=pred_star.device
            )

            for diff_idx, (min_star, max_star) in self.star_ranges.items():
                mask = difficulty == diff_idx
                is_right_censored[mask] = target_star[mask] >= max_star
                is_left_censored[mask] = target_star[mask] <= min_star

        # Compute losses per sample
        losses = torch.zeros_like(pred_star)

        # Right-censored: only penalize if pred < target
        right_mask = is_right_censored
        if right_mask.any():
            shortfall = F.relu(target_star[right_mask] - pred_star[right_mask])
            losses[right_mask] = shortfall

        # Left-censored: only penalize if pred > target
        left_mask = is_left_censored
        if left_mask.any():
            overshoot = F.relu(pred_star[left_mask] - target_star[left_mask])
            losses[left_mask] = overshoot

        # Uncensored: standard loss
        uncensored_mask = ~(is_right_censored | is_left_censored)
        if uncensored_mask.any():
            losses[uncensored_mask] = self._uncensored_loss(
                pred_star[uncensored_mask],
                target_star[uncensored_mask],
            )

        return losses.mean()

    def compute_censoring_metrics(
        self,
        pred_star: torch.Tensor,
        target_star: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> dict:
        """
        Compute censoring-related metrics.

        Returns:
            Dict with violation rates and shortfall/overshoot stats
        """
        metrics = {}

        for diff_idx, (min_star, max_star) in self.star_ranges.items():
            mask = difficulty == diff_idx
            if not mask.any():
                continue

            preds = pred_star[mask]
            targets = target_star[mask]

            # Right-censored samples (at max)
            right_mask = targets >= max_star
            if right_mask.any():
                right_preds = preds[right_mask]
                violation_rate = (right_preds < max_star).float().mean().item()
                mean_shortfall = F.relu(max_star - right_preds).mean().item()

                metrics[f"right_violation_rate_{diff_idx}"] = violation_rate
                metrics[f"mean_shortfall_{diff_idx}"] = mean_shortfall

            # Left-censored samples (at min)
            left_mask = targets <= min_star
            if left_mask.any():
                left_preds = preds[left_mask]
                violation_rate = (left_preds > min_star).float().mean().item()
                mean_overshoot = F.relu(left_preds - min_star).mean().item()

                metrics[f"left_violation_rate_{diff_idx}"] = violation_rate
                metrics[f"mean_overshoot_{diff_idx}"] = mean_overshoot

        return metrics


class TotalLoss(nn.Module):
    """
    Multi-task loss combiner for difficulty estimation.

    Combines:
    - Classification loss (difficulty prediction)
    - Censored star regression loss
    - Within-song ranking loss (monotonicity)

    Supports curriculum learning with schedulable weights.
    Note: When merge_ura_oni=True, ura (4) and oni (3) are treated as the same class.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_star: float = 1.0,
        lambda_rank: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ranking_margin: float = 0.5,
        star_loss_type: str = "huber",
        merge_ura_oni: bool = True,
    ):
        """
        Args:
            lambda_cls: Weight for classification loss
            lambda_star: Weight for star regression loss
            lambda_rank: Weight for ranking loss
            class_weights: Optional class weights for classification
            ranking_margin: Margin for ranking hinge loss
            star_loss_type: Loss type for star regression
            merge_ura_oni: If True, treat ura (4) as oni (3) for classification
        """
        super().__init__()

        self.lambda_cls = lambda_cls
        self.lambda_star = lambda_star
        self.lambda_rank = lambda_rank
        self.merge_ura_oni = merge_ura_oni

        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Star regression loss
        self.star_loss = CensoredRegressionLoss(uncensored_loss=star_loss_type)

        # Ranking loss
        self.rank_loss = WithinSongRankingLoss(margin=ranking_margin)

    def set_weights(
        self,
        lambda_cls: Optional[float] = None,
        lambda_star: Optional[float] = None,
        lambda_rank: Optional[float] = None,
    ):
        """Update loss weights (for curriculum learning)."""
        if lambda_cls is not None:
            self.lambda_cls = lambda_cls
        if lambda_star is not None:
            self.lambda_star = lambda_star
        if lambda_rank is not None:
            self.lambda_rank = lambda_rank

    def forward(
        self,
        difficulty_logits: torch.Tensor,
        pred_star: torch.Tensor,
        target_difficulty: torch.Tensor,
        target_star: torch.Tensor,
        is_right_censored: Optional[torch.Tensor] = None,
        is_left_censored: Optional[torch.Tensor] = None,
        ranking_pairs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total loss with breakdown.

        Args:
            difficulty_logits: [batch, n_classes] difficulty predictions
            pred_star: [batch] predicted star ratings
            target_difficulty: [batch] target difficulty classes
            target_star: [batch] target star labels
            is_right_censored: [batch] right-censoring flags
            is_left_censored: [batch] left-censoring flags
            ranking_pairs: Optional (s_easier, s_harder) for ranking loss

        Returns:
            Dict with total loss and breakdown:
                - "total": Combined weighted loss
                - "cls": Classification loss
                - "star": Star regression loss
                - "rank": Ranking loss (if pairs provided)
        """
        losses = {}

        # Classification loss
        # Merge ura (4) and oni (3) if enabled
        if self.merge_ura_oni:
            # Merge target: map ura (class 4) to oni (class 3)
            target_difficulty_merged = target_difficulty.clone()
            target_difficulty_merged[target_difficulty_merged == 4] = 3

            # Correct merging: use logsumexp in log-probability space
            # This correctly computes P(oni OR ura) = P(oni) + P(ura)
            log_probs = F.log_softmax(difficulty_logits, dim=-1)  # [batch, 5]
            log_probs_merged = log_probs[:, :4].clone()  # [batch, 4]
            # logsumexp(log P(oni), log P(ura)) = log(P(oni) + P(ura))
            log_probs_merged[:, 3] = torch.logsumexp(log_probs[:, 3:5], dim=-1)

            cls_loss = F.nll_loss(
                log_probs_merged,
                target_difficulty_merged,
                weight=self.cls_loss.weight,
            )
        else:
            cls_loss = self.cls_loss(difficulty_logits, target_difficulty)
        losses["cls"] = cls_loss

        # Star regression loss
        star_loss = self.star_loss(
            pred_star,
            target_star,
            target_difficulty,
            is_right_censored,
            is_left_censored,
        )
        losses["star"] = star_loss

        # Ranking loss (if pairs provided)
        if ranking_pairs is not None:
            s_easier, s_harder = ranking_pairs
            rank_loss = self.rank_loss(s_easier, s_harder)
            losses["rank"] = rank_loss
        else:
            rank_loss = torch.tensor(0.0, device=pred_star.device)
            losses["rank"] = rank_loss

        # Combine with weights
        total = (
            self.lambda_cls * cls_loss
            + self.lambda_star * star_loss
            + self.lambda_rank * rank_loss
        )
        losses["total"] = total

        return losses


class CurriculumScheduler:
    """
    Scheduler for curriculum learning of loss weights.

    Early training: focus on classification (coarse alignment)
    Later training: increase ranking + star loss (fine-grained)
    """

    def __init__(
        self,
        total_steps: int,
        warmup_fraction: float = 0.2,
        cls_start: float = 2.0,
        cls_end: float = 0.5,
        rank_start: float = 0.1,
        rank_end: float = 1.5,
        star_start: float = 0.5,
        star_end: float = 1.5,
    ):
        """
        Args:
            total_steps: Total training steps
            warmup_fraction: Fraction of training for warmup
            *_start/*_end: Start and end values for each loss weight
        """
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)

        self.cls_start = cls_start
        self.cls_end = cls_end
        self.rank_start = rank_start
        self.rank_end = rank_end
        self.star_start = star_start
        self.star_end = star_end

    def get_weights(self, step: int) -> dict[str, float]:
        """
        Get loss weights for current step.

        Returns:
            Dict with lambda_cls, lambda_star, lambda_rank
        """
        # Smooth interpolation over entire training
        if self.total_steps > 0:
            t = min(1.0, step / self.total_steps)
        else:
            t = 1.0

        # Linear interpolation from start to end
        lambda_cls = self.cls_start + t * (self.cls_end - self.cls_start)
        lambda_rank = self.rank_start + t * (self.rank_end - self.rank_start)
        lambda_star = self.star_start + t * (self.star_end - self.star_start)

        return {
            "lambda_cls": lambda_cls,
            "lambda_star": lambda_star,
            "lambda_rank": lambda_rank,
        }
