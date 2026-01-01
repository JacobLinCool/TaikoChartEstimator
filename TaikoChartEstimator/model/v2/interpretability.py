"""
Interpretability Module for TaikoChartEstimator - Version 2

Provides gradient-based feature attribution for BOTH continuous features
and discrete note patterns, relying entirely on model-learned importance.

Features:
- Continuous feature attribution (Speed, Density, Rhythm)
- Discrete embedding attribution (Model's sensitivity to Note Types)
- Interpretable difficulty breakdown derived from gradients
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FeatureContribution:
    """Contribution of a feature group to the prediction."""

    name: str
    value: float  # Absolute contribution magnitude
    direction: str  # "increases" or "decreases" difficulty (inferred)
    components: dict[str, float] = field(
        default_factory=dict
    )  # Individual feature contributions


@dataclass
class InterpretabilityReport:
    """
    Comprehensive interpretability report for a prediction.

    Attributes:
        arm_strength_required: Inferred from Density feature attribution
        visual_analysis_speed: Inferred from Speed feature attribution
        rhythm_complexity: Inferred from Rhythm feature attribution
        pattern_importance: Model's attribution to different note types (Don/Ka/Big/etc)
        per_instance_scores: Per-instance difficulty scores for visualization
        # feature_attributions: Raw gradient-based attributions (internal use)
    """

    arm_strength_required: FeatureContribution
    visual_analysis_speed: FeatureContribution
    rhythm_complexity: FeatureContribution
    pattern_importance: Optional[dict[str, float]] = None
    per_instance_scores: Optional[torch.Tensor] = None


class FeatureAttributor(nn.Module):
    """
    Gradient-based sensitivity analysis for continuous features.

    Computes ∂raw_score / ∂feature for each continuous feature.
    """

    # Feature indices in the token tensor (after note_type)
    FEATURE_NAMES = ["beat_position", "duration", "bpm", "scroll", "local_density"]
    FEATURE_GROUPS = {
        "speed": ["bpm", "scroll"],
        "density": ["local_density"],
        "rhythm": ["beat_position", "duration"],
    }

    def __init__(self):
        super().__init__()

    @torch.enable_grad()
    def compute_continuous_attributions(
        self,
        model: nn.Module,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
        instance_counts: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Compute gradient-based feature attributions for continuous features.
        """
        # Ensure gradients are enabled for input (only continuous part)
        # instances: [batch, n_inst, seq_len, 6]
        # We need to clone and detach to set leaf requirements, but instances is usually leaf.
        # To be safe, we create a new tensor for the continuous part that requires grad.

        # Split input
        note_types = instances[..., 0:1]  # [..., 1]
        continuous = (
            instances[..., 1:].clone().detach().requires_grad_(True)
        )  # [..., 5]

        # Re-assemble for model (we need to pass this exact tensor to get grads on it)
        # Note: We can't easily re-assemble and pass to model(x) because model expects one tensor.
        # But we can reconstruct:
        model_input = torch.cat([note_types, continuous], dim=-1)

        # Forward pass
        output = model(
            model_input, instance_masks, instance_counts, return_attention=False
        )
        raw_score = output.raw_score

        # Backward
        loss = raw_score.sum()
        loss.backward()

        # Get gradients from the continuous tensor we created
        # Note: model_input was created by concatenation, so it's not a leaf.
        # But `continuous` is a leaf. Gradients propagate through cat.
        grads = continuous.grad  # [batch, n_inst, seq_len, 5]

        attributions = {}
        if grads is not None:
            for i, name in enumerate(self.FEATURE_NAMES):
                # Mean absolute gradient * feature value (optional, here just Grad)
                # Using Grad * Input (saliency) is often better for magnitude
                # attr = (grads[..., i] * continuous[..., i]).abs().mean().item()
                # Let's use simple Gradient Magnitude for sensitivity
                attr = grads[..., i].abs().mean().item()
                attributions[name] = attr

        return attributions

    def group_contributions(
        self, attributions: dict[str, float]
    ) -> dict[str, FeatureContribution]:
        """Group continuous attributions into categories."""
        contributions = {}
        for group, feats in self.FEATURE_GROUPS.items():
            total = 0.0
            comps = {}
            for f in feats:
                val = attributions.get(f, 0.0)
                comps[f] = val
                total += val

            # Direction is hard to infer purely from gradient magnitude (always positive).
            # We assume positive contribution to difficulty for now or heuristic.
            contributions[group] = FeatureContribution(
                name=group, value=total, direction="increases", components=comps
            )
        return contributions


class DiscretePatternAttributor:
    """
    Gradient-based attribution for Discrete Note Types.
    Uses Input * Gradient on the Embedding outputs to measure importance.
    """

    # Note type definitions for grouping
    DON_TYPES = {0, 2}
    KA_TYPES = {1, 3}
    BIG_TYPES = {2, 3, 5}
    LONG_TYPES = {4, 5, 6, 7}
    TYPE_NAMES = {
        0: "Don",
        1: "Ka",
        2: "DonBig",
        3: "KaBig",
        4: "Roll",
        5: "RollBig",
        6: "Balloon",
        7: "BalloonAlt",
        8: "None",
        9: "Pad",
    }

    def compute_pattern_importance(
        self,
        model: nn.Module,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
        instance_counts: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Compute importance of different note types using Embedding Gradients.
        """
        # Hook into the embedding layer
        embeddings_captured = []

        def hook_fn(module, input, output):
            # output is [batch, seq_len, d_model] or similar depending on encoder
            # We retain grad to access it after backward
            output.retain_grad()
            embeddings_captured.append(output)

        # Locate embedding layer
        # Assumes model.instance_encoder.type_embedding exists
        if hasattr(model.instance_encoder, "type_embedding"):
            embed_layer = model.instance_encoder.type_embedding
        else:
            return {}  # Can't find embedding layer

        handle = embed_layer.register_forward_hook(hook_fn)

        try:
            # Forward pass
            model.zero_grad()
            output = model(
                instances, instance_masks, instance_counts, return_attention=False
            )
            score = output.raw_score.sum()

            # Backward to get gradients on embeddings
            score.backward()

            if not embeddings_captured:
                return {}

            # embedding_tensor: [batch, ..., d_model]
            # embedding_grad: [batch, ..., d_model]
            embedding_tensor = embeddings_captured[0]
            embedding_grad = embedding_tensor.grad

            if embedding_grad is None:
                return {}

            # Compute Saliency: |Grad * Input| (dot product sum over embedding dim)
            # saliency per token: [batch, n_inst, seq_len]
            # Since transformer flattens batch/inst often, input might be flattened.
            # let's match shapes.

            # The encoder merges batch*instances.
            # embedded tensor shape likely: [batch*n_inst, seq_len, d_model]

            # Attribution per token = sum(|grad * val|) or just |grad|?
            # "Input * Gradient" is standard.
            token_attribution = (embedding_grad * embedding_tensor).sum(dim=-1).abs()
            # shape: [total_sequences, seq_len]

            # We need to map these back to the note types
            # instances Input: [batch, n_inst, seq_len, 6]
            # We need to flatten instances to match hook output
            # Note types are at index 0
            all_note_types = (
                instances[..., 0].long().flatten(0, 1)
            )  # [batch*n_inst, seq_len]
            all_masks = instance_masks.flatten(0, 1)  # [batch*n_inst, seq_len]

            # Verify shapes match
            if token_attribution.shape[:2] != all_note_types.shape[:2]:
                # If shapes don't match (e.g. TCN might reshape), abort or try to handle
                return {"error": "Shape mismatch in attribution analysis"}

            # Aggregate by type
            type_importance = {}
            total_importance = 0.0

            # Iterate through unique types present
            unique_types = torch.unique(all_note_types)

            for tid in unique_types:
                tid_val = tid.item()
                if tid_val == 9:
                    continue  # Skip Pad

                mask = (all_note_types == tid) & (all_masks > 0)
                if mask.sum() == 0:
                    continue

                # Sum attribution for this type
                importance = token_attribution[mask].sum().item()
                count = mask.sum().item()

                # Average importance per occurrence? Or total?
                # User wants "Explanation". "Dons are very important" vs "There are many Dons".
                # Total contribution to score is usually sum.
                type_importance[self.TYPE_NAMES.get(tid_val, str(tid_val))] = importance
                total_importance += importance

            # Normalize to percent? or keep raw magnitude?
            # Creating summary groups
            summary = {
                "Don_Importance": 0.0,
                "Ka_Importance": 0.0,
                "Big_Note_Importance": 0.0,
                "Long_Note_Importance": 0.0,
            }

            for t_name, score in type_importance.items():
                if "Don" in t_name:
                    summary["Don_Importance"] += score
                if "Ka" in t_name:
                    summary["Ka_Importance"] += score
                if "Big" in t_name:
                    summary["Big_Note_Importance"] += score
                if "Roll" in t_name or "Balloon" in t_name:
                    summary["Long_Note_Importance"] += score

            # Normalize summary by total to see relative focus
            if total_importance > 0:
                final_summary = {k: v / total_importance for k, v in summary.items()}
                # Also include raw top types
                # final_summary.update(type_importance)
            else:
                final_summary = summary

            return final_summary

        finally:
            handle.remove()


class ChartInterpreter:
    """
    Main interface for chart interpretability analysis.

    Uses mostly gradient-based attribution to let the model explain itself.
    """

    def __init__(self):
        self.feature_attributor = FeatureAttributor()
        self.pattern_attributor = DiscretePatternAttributor()

    def analyze(
        self,
        model: nn.Module,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
        instance_counts: Optional[torch.Tensor] = None,
        compute_gradients: bool = True,
    ) -> InterpretabilityReport:
        """
        Generate interpretability report.
        """
        if not compute_gradients:
            # Return empty/dummy report if gradients disabled (e.g. inference only optimization)
            return InterpretabilityReport(
                arm_strength_required=FeatureContribution("density", 0, "neutral"),
                visual_analysis_speed=FeatureContribution("speed", 0, "neutral"),
                rhythm_complexity=FeatureContribution("rhythm", 0, "neutral"),
            )

        # 1. Continuous Features Attribution
        cont_attrs = self.feature_attributor.compute_continuous_attributions(
            model, instances, instance_masks, instance_counts
        )
        cont_groups = self.feature_attributor.group_contributions(cont_attrs)

        # 2. Discrete Pattern Attribution (Model learned importance)
        pattern_importance = self.pattern_attributor.compute_pattern_importance(
            model, instances, instance_masks, instance_counts
        )

        # 3. Per-instance scores
        model.eval()
        with torch.no_grad():
            output = model(
                instances, instance_masks, instance_counts, return_attention=True
            )
            if hasattr(model, "get_instance_scores"):
                _, per_instance_star = model.get_instance_scores(
                    output.instance_embeddings
                )
            else:
                per_instance_star = None

        return InterpretabilityReport(
            arm_strength_required=cont_groups.get("density"),
            visual_analysis_speed=cont_groups.get("speed"),
            rhythm_complexity=cont_groups.get("rhythm"),
            pattern_importance=pattern_importance,
            per_instance_scores=per_instance_star,
        )

    def format_report(self, report: InterpretabilityReport) -> str:
        """Format report as human-readable text."""
        lines = ["=== Chart Difficulty Analysis (Model Learned) ===", ""]

        # Continuous Groups
        for subgroup, title in [
            (report.arm_strength_required, "Arm Strength (Density)"),
            (report.visual_analysis_speed, "Visual Speed"),
            (report.rhythm_complexity, "Rhythm Complexity"),
        ]:
            if subgroup:
                lines.append(f"{title}: {subgroup.value:.4f}")
                for k, v in subgroup.components.items():
                    lines.append(f"  - {k}: {v:.4f}")
            lines.append("")

        # Pattern Importance
        if report.pattern_importance:
            lines.append("Model Pattern Attention (Relative Importance):")
            for k, v in report.pattern_importance.items():
                lines.append(f"  - {k}: {v:.1%}")

        return "\n".join(lines)
