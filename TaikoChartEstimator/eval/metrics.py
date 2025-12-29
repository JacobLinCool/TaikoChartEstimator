"""
Evaluation Metrics for TaikoChartEstimator

Comprehensive metrics covering:
- Difficulty classification
- Star rating regression (with censoring awareness)
- Monotonicity constraints
- 10-star decompression
- MIL attention health
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from ..constants import STAR_RANGES_BY_ID


@dataclass
class DifficultyMetrics:
    """
    Metrics for difficulty classification (easy/normal/hard/oni/ura).

    Includes ordinal-aware metrics since difficulties are ordered.
    Note: ura (4) and oni (3) are treated as the same class for metrics.
    """

    merge_ura_oni: bool = True  # Treat ura and oni as the same class

    def _merge_classes(self, arr: np.ndarray) -> np.ndarray:
        """Merge ura (4) into oni (3) class."""
        if self.merge_ura_oni:
            arr = arr.copy()
            arr[arr == 4] = 3  # Map ura -> oni
        return arr

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted difficulty class indices [N]
            targets: True difficulty class indices [N]

        Returns:
            Dict with all metrics
        """
        metrics = {}

        # Merge ura and oni classes if enabled
        predictions = self._merge_classes(predictions)
        targets = self._merge_classes(targets)

        # Standard classification metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["balanced_accuracy"] = balanced_accuracy_score(targets, predictions)
        metrics["macro_f1"] = f1_score(targets, predictions, average="macro")
        metrics["weighted_f1"] = f1_score(targets, predictions, average="weighted")

        # Per-class F1 (4 classes when merged: easy, normal, hard, oni/ura)
        per_class_f1 = f1_score(targets, predictions, average=None)
        if self.merge_ura_oni:
            class_names = ["easy", "normal", "hard", "oni_ura"]
        else:
            class_names = ["easy", "normal", "hard", "oni", "ura"]
        for i, name in enumerate(class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = per_class_f1[i]

        # Ordinal-aware metrics (difficulties are ordered)
        abs_diff = np.abs(predictions - targets)
        metrics["mean_absolute_error_ordinal"] = abs_diff.mean()
        metrics["plus_minus_1_accuracy"] = (abs_diff <= 1).mean()
        metrics["plus_minus_2_accuracy"] = (abs_diff <= 2).mean()

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(targets, predictions)

        return metrics


@dataclass
class StarMetrics:
    """
    Metrics for star rating prediction with censoring awareness.

    Separates metrics for:
    - Uncensored samples (true regression quality)
    - Right-censored samples (10-star boundary)
    - Left-censored samples (1-star boundary)
    """

    star_ranges: dict = field(default_factory=lambda: STAR_RANGES_BY_ID.copy())

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        difficulties: np.ndarray,
        is_right_censored: Optional[np.ndarray] = None,
        is_left_censored: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute star regression metrics.

        Args:
            predictions: Predicted star ratings [N]
            targets: Target star labels [N]
            difficulties: Difficulty class indices [N]
            is_right_censored: Boolean mask for right-censored samples
            is_left_censored: Boolean mask for left-censored samples

        Returns:
            Dict with all metrics
        """
        metrics = {}

        # Auto-detect censoring if not provided
        if is_right_censored is None or is_left_censored is None:
            is_right_censored = np.zeros(len(predictions), dtype=bool)
            is_left_censored = np.zeros(len(predictions), dtype=bool)

            for diff_idx, (min_star, max_star) in self.star_ranges.items():
                mask = difficulties == diff_idx
                is_right_censored[mask] = targets[mask] >= max_star
                is_left_censored[mask] = targets[mask] <= min_star

        # Overall metrics
        metrics["mae"] = np.abs(predictions - targets).mean()
        metrics["rmse"] = np.sqrt(((predictions - targets) ** 2).mean())

        if len(predictions) > 1:
            rho, p_value = spearmanr(predictions, targets)
            metrics["spearman_rho"] = rho
            metrics["spearman_pvalue"] = p_value
        else:
            metrics["spearman_rho"] = 0.0
            metrics["spearman_pvalue"] = 1.0

        # Uncensored samples: true regression quality
        uncensored_mask = ~(is_right_censored | is_left_censored)
        if uncensored_mask.sum() > 0:
            uncensored_preds = predictions[uncensored_mask]
            uncensored_targets = targets[uncensored_mask]

            metrics["mae_uncensored"] = np.abs(
                uncensored_preds - uncensored_targets
            ).mean()
            metrics["rmse_uncensored"] = np.sqrt(
                ((uncensored_preds - uncensored_targets) ** 2).mean()
            )

            if len(uncensored_preds) > 1:
                rho, _ = spearmanr(uncensored_preds, uncensored_targets)
                metrics["spearman_rho_uncensored"] = rho
            else:
                metrics["spearman_rho_uncensored"] = 0.0
        else:
            metrics["mae_uncensored"] = 0.0
            metrics["rmse_uncensored"] = 0.0
            metrics["spearman_rho_uncensored"] = 0.0

        # Right-censored (at max star): check violation
        if is_right_censored.sum() > 0:
            right_preds = predictions[is_right_censored]
            right_targets = targets[is_right_censored]

            # Violation: prediction below the max star bound
            violation_mask = right_preds < right_targets
            metrics["right_censor_violation_rate"] = violation_mask.mean()

            if violation_mask.sum() > 0:
                metrics["right_censor_mean_shortfall"] = (
                    right_targets[violation_mask] - right_preds[violation_mask]
                ).mean()
            else:
                metrics["right_censor_mean_shortfall"] = 0.0

            metrics["right_censor_count"] = is_right_censored.sum()
        else:
            metrics["right_censor_violation_rate"] = 0.0
            metrics["right_censor_mean_shortfall"] = 0.0
            metrics["right_censor_count"] = 0

        # Left-censored (at min star): check violation
        if is_left_censored.sum() > 0:
            left_preds = predictions[is_left_censored]
            left_targets = targets[is_left_censored]

            # Violation: prediction above the min star bound
            violation_mask = left_preds > left_targets
            metrics["left_censor_violation_rate"] = violation_mask.mean()

            if violation_mask.sum() > 0:
                metrics["left_censor_mean_overshoot"] = (
                    left_preds[violation_mask] - left_targets[violation_mask]
                ).mean()
            else:
                metrics["left_censor_mean_overshoot"] = 0.0

            metrics["left_censor_count"] = is_left_censored.sum()
        else:
            metrics["left_censor_violation_rate"] = 0.0
            metrics["left_censor_mean_overshoot"] = 0.0
            metrics["left_censor_count"] = 0

        return metrics


@dataclass
class MonotonicityMetrics:
    """
    Metrics for within-song monotonicity constraint.

    Checks that harder difficulties have higher scores/stars
    within the same song.
    """

    difficulty_order: dict = field(
        default_factory=lambda: {
            "easy": 0,
            "Easy": 0,
            "normal": 1,
            "Normal": 1,
            "hard": 2,
            "Hard": 2,
            "oni": 3,
            "Oni": 3,
            "ura": 4,
            "Ura": 4,
        }
    )

    def compute(
        self,
        raw_scores: np.ndarray,
        song_ids: list[str],
        difficulties: list[str],
    ) -> dict:
        """
        Compute monotonicity metrics.

        Args:
            raw_scores: Raw difficulty scores [N]
            song_ids: Song identifiers
            difficulties: Difficulty names

        Returns:
            Dict with metrics
        """
        metrics = {}

        # Group by song
        song_groups: dict[str, list] = {}
        for i, song_id in enumerate(song_ids):
            if song_id not in song_groups:
                song_groups[song_id] = []
            song_groups[song_id].append(
                {
                    "idx": i,
                    "difficulty": difficulties[i],
                    "score": raw_scores[i],
                }
            )

        # Count violations
        n_violations = 0
        n_pairs = 0
        violation_margins = []
        per_song_kendall_tau = []

        for song_id, charts in song_groups.items():
            if len(charts) < 2:
                continue

            # Sort by difficulty order
            sorted_charts = sorted(
                charts, key=lambda c: self.difficulty_order.get(c["difficulty"], 0)
            )

            # Check adjacent pairs
            for i in range(len(sorted_charts) - 1):
                n_pairs += 1
                score_easier = sorted_charts[i]["score"]
                score_harder = sorted_charts[i + 1]["score"]

                if score_easier >= score_harder:
                    n_violations += 1
                    violation_margins.append(score_easier - score_harder)

            # Compute Kendall's tau within song
            if len(sorted_charts) >= 2:
                actual_scores = [c["score"] for c in sorted_charts]
                expected_ranks = list(range(len(sorted_charts)))

                tau, _ = kendalltau(actual_scores, expected_ranks)
                if not np.isnan(tau):
                    per_song_kendall_tau.append(tau)

        # Aggregate metrics
        metrics["n_pairs"] = n_pairs
        metrics["n_violations"] = n_violations
        metrics["violation_rate"] = n_violations / n_pairs if n_pairs > 0 else 0.0

        if violation_margins:
            metrics["mean_violation_margin"] = np.mean(violation_margins)
            metrics["max_violation_margin"] = np.max(violation_margins)
        else:
            metrics["mean_violation_margin"] = 0.0
            metrics["max_violation_margin"] = 0.0

        if per_song_kendall_tau:
            metrics["mean_kendall_tau_within_song"] = np.mean(per_song_kendall_tau)
            metrics["min_kendall_tau_within_song"] = np.min(per_song_kendall_tau)
        else:
            metrics["mean_kendall_tau_within_song"] = 0.0
            metrics["min_kendall_tau_within_song"] = 0.0

        return metrics


@dataclass
class DecompressionMetrics:
    """
    Metrics for 10-star decompression.

    Checks if the model learns to distinguish between different
    10-star charts (which vary widely in actual difficulty).
    """

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        difficulties: np.ndarray,
    ) -> dict:
        """
        Compute decompression metrics for max-star samples.

        Args:
            predictions: Predicted star ratings (can exceed range)
            targets: Target star labels
            difficulties: Difficulty indices

        Returns:
            Dict with metrics
        """
        metrics = {}

        # Star ranges per difficulty
        max_stars = {0: 5, 1: 7, 2: 8, 3: 10, 4: 10}

        for diff_idx, max_star in max_stars.items():
            mask = (difficulties == diff_idx) & (targets >= max_star)

            if mask.sum() < 2:
                continue

            preds_at_max = predictions[mask]
            diff_name = ["easy", "normal", "hard", "oni", "ura"][diff_idx]

            # Spread of predictions
            metrics[f"std_{diff_name}_max"] = preds_at_max.std()

            # Percentile gaps
            if len(preds_at_max) >= 10:
                p50 = np.percentile(preds_at_max, 50)
                p90 = np.percentile(preds_at_max, 90)
                p99 = np.percentile(preds_at_max, 99)

                metrics[f"p90_p50_{diff_name}"] = p90 - p50
                metrics[f"p99_p90_{diff_name}"] = p99 - p90

            # Range
            metrics[f"range_{diff_name}_max"] = preds_at_max.max() - preds_at_max.min()
            metrics[f"n_samples_{diff_name}_max"] = mask.sum()

        # Overall 10-star decompression (oni + ura combined)
        max_10_mask = (targets >= 10) & ((difficulties == 3) | (difficulties == 4))
        if max_10_mask.sum() >= 2:
            preds_10star = predictions[max_10_mask]

            metrics["std_10star"] = preds_10star.std()
            metrics["range_10star"] = preds_10star.max() - preds_10star.min()
            metrics["n_samples_10star"] = max_10_mask.sum()

            if len(preds_10star) >= 10:
                metrics["p90_p50_10star"] = np.percentile(
                    preds_10star, 90
                ) - np.percentile(preds_10star, 50)
                metrics["p99_p90_10star"] = np.percentile(
                    preds_10star, 99
                ) - np.percentile(preds_10star, 90)

        return metrics


@dataclass
class MILHealthMetrics:
    """
    Metrics for MIL attention health.

    Monitors attention distribution to detect collapse
    (model focusing on too few instances).
    """

    def compute(
        self,
        attention_weights: list[np.ndarray],
        instance_counts: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute MIL attention health metrics.

        Args:
            attention_weights: List of attention weight arrays, each with shape
                               [N_instances] where N_instances can vary per sample.
            instance_counts: Number of valid instances per sample (optional,
                             used for health ratio calculation)

        Returns:
            Dict with metrics
        """
        metrics = {}

        if not attention_weights:
            return metrics

        n_samples = len(attention_weights)

        # Attention entropy per sample
        # Higher entropy = more distributed attention (good for MIL)
        entropies = []
        effective_ns = []
        top5_masses = []
        actual_instance_counts = []

        for i, attn in enumerate(attention_weights):
            if len(attn) == 0:
                continue

            # Use instance_counts to mask if provided, otherwise use full attention
            if instance_counts is not None:
                n_valid = int(instance_counts[i])
                attn = attn[:n_valid]

            if len(attn) == 0:
                continue

            actual_instance_counts.append(len(attn))

            # Normalize to sum to 1
            attn = attn / (attn.sum() + 1e-8)

            # Entropy
            entropy = -np.sum(attn * np.log(attn + 1e-8))
            entropies.append(entropy)

            # Effective number of instances (inverse of concentration)
            effective_n = 1.0 / (np.sum(attn**2) + 1e-8)
            effective_ns.append(effective_n)

            # Top-5% mass
            k = max(1, int(len(attn) * 0.05))
            top5_mass = np.sort(attn)[-k:].sum()
            top5_masses.append(top5_mass)

        if entropies:
            metrics["mean_attention_entropy"] = np.mean(entropies)
            metrics["min_attention_entropy"] = np.min(entropies)
            metrics["std_attention_entropy"] = np.std(entropies)

        if effective_ns:
            metrics["mean_effective_instances"] = np.mean(effective_ns)
            metrics["min_effective_instances"] = np.min(effective_ns)

        if top5_masses:
            metrics["mean_top5_mass"] = np.mean(top5_masses)
            metrics["max_top5_mass"] = np.max(top5_masses)

        # Health assessment
        # Collapse warning if too few effective instances
        if effective_ns and actual_instance_counts:
            collapse_ratio = np.mean(effective_ns) / np.mean(actual_instance_counts)
            metrics["health_ratio"] = collapse_ratio
            metrics["attention_collapse_warning"] = (
                collapse_ratio < 0.1
            )  # Less than 10% of instances used

        return metrics
