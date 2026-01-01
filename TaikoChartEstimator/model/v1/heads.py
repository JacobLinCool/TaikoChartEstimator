"""
Output Heads for Taiko Chart Estimation

Three heads for multi-task learning:
- Head A: Raw difficulty score (unbounded)
- Head B: Difficulty classification (4-5 classes)
- Head C: Monotonic star calibration
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RawScoreHead(nn.Module):
    """
    Head A: Unbounded raw difficulty score.

    Outputs s ∈ ℝ, the "true" continuous difficulty scale
    before mapping to display star ratings.
    """

    def __init__(
        self,
        d_input: int = 512,
        d_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # Initialize to output reasonable range (~1-10)
        self._init_weights()

    def _init_weights(self):
        """Initialize to output values centered around 5."""
        with torch.no_grad():
            # Bias the final layer to output ~5
            self.mlp[-1].bias.fill_(5.0)
            self.mlp[-1].weight.fill_(0.01)

    def forward(self, bag_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bag_embedding: [batch, d_input]

        Returns:
            raw_score: [batch] unbounded difficulty score
        """
        return self.mlp(bag_embedding).squeeze(-1)


class DifficultyClassifier(nn.Module):
    """
    Head B: Difficulty classification.

    Predicts difficulty class: easy, normal, hard, oni, ura (5 classes)
    or merged oni_ura (4 classes).
    """

    def __init__(
        self,
        d_input: int = 512,
        n_classes: int = 5,
        d_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_classes = n_classes

        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, bag_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bag_embedding: [batch, d_input]

        Returns:
            logits: [batch, n_classes] classification logits
        """
        return self.mlp(bag_embedding)

    def predict(self, bag_embedding: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(bag_embedding)
        return logits.argmax(dim=-1)


class MonotonicSpline(nn.Module):
    """
    Monotonic spline for mapping raw score to star rating.

    Uses I-splines (integrated B-splines) to guarantee monotonicity.
    Learnable coefficients are constrained to be positive.
    """

    def __init__(
        self,
        n_knots: int = 8,
        input_range: tuple[float, float] = (0, 15),
        output_range: tuple[float, float] = (1, 10),
    ):
        super().__init__()

        self.n_knots = n_knots
        self.input_range = input_range
        self.output_range = output_range

        # Knot positions (fixed)
        knots = torch.linspace(input_range[0], input_range[1], n_knots)
        self.register_buffer("knots", knots)

        # Learnable positive coefficients (using softplus for positivity)
        self.raw_coefficients = nn.Parameter(torch.ones(n_knots))

        # Learnable offset
        self.offset = nn.Parameter(torch.tensor(float(output_range[0])))

    def _compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute I-spline basis functions with clamping for stability."""
        # Clamp input to reasonable range to prevent output explosion
        x_clamped = x.clamp(self.input_range[0], self.input_range[1])
        x_clamped = x_clamped.unsqueeze(-1)  # [batch, 1]
        knots = self.knots.unsqueeze(0)  # [1, n_knots]

        # Compute distance to each knot
        diff = x_clamped - knots  # [batch, n_knots]

        # ReLU with cap to prevent unbounded growth
        # Cap at input_range width for reasonable behavior
        max_value = self.input_range[1] - self.input_range[0]
        basis = F.relu(diff).clamp(max=max_value)  # [batch, n_knots]

        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map raw score to star rating (monotonically).

        Args:
            x: Raw scores [batch]

        Returns:
            Star ratings [batch]
        """
        # Ensure positive coefficients
        coefficients = F.softplus(self.raw_coefficients)

        # Normalize coefficients to control output scale
        coefficients = coefficients / coefficients.sum()
        scale = self.output_range[1] - self.output_range[0]
        coefficients = coefficients * scale

        # Compute basis
        basis = self._compute_basis(x)  # [batch, n_knots]

        # Weighted sum
        output = (basis * coefficients).sum(dim=-1) + self.offset

        return output


class MonotonicMLP(nn.Module):
    """
    Monotonic MLP using positive weight constraints.

    Ensures f(x1) >= f(x2) whenever x1 >= x2 by constraining
    all weights to be positive and using monotonic activations.
    """

    def __init__(
        self,
        d_hidden: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        layers = []
        in_dim = 1

        for i in range(n_layers):
            out_dim = d_hidden if i < n_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.Softplus())  # Monotonic activation
            in_dim = out_dim

        self.layers = nn.ModuleList(
            [layer for layer in layers if isinstance(layer, nn.Linear)]
        )
        self.activations = [nn.Softplus() for _ in range(n_layers - 1)] + [
            nn.Identity()
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw scores [batch]

        Returns:
            Calibrated scores [batch]
        """
        out = x.unsqueeze(-1)  # [batch, 1]

        for layer, activation in zip(self.layers, self.activations):
            # Apply absolute value to weights for monotonicity
            weight = layer.weight.abs()
            out = F.linear(out, weight, layer.bias)
            out = activation(out)

        return out.squeeze(-1)


class MonotonicCalibrator(nn.Module):
    """
    Head C: Monotonic calibration from raw score to star rating.

    Maintains separate calibrators per difficulty level, since
    the star ranges differ (easy: 1-5, normal: 1-7, etc.)

    Guarantees:
    - Output is monotonically increasing with input
    - Can output values outside the nominal range (for decompression)
    """

    def __init__(
        self,
        method: str = "spline",  # "spline" or "mlp"
        n_difficulties: int = 5,
        star_ranges: Optional[dict] = None,
    ):
        """
        Args:
            method: Calibration method ("spline" or "mlp")
            n_difficulties: Number of difficulty classes
            star_ranges: Dict mapping difficulty index to (min, max) star range
        """
        super().__init__()

        self.method = method
        self.n_difficulties = n_difficulties

        # Default star ranges per difficulty
        if star_ranges is None:
            star_ranges = {
                0: (1, 5),  # easy
                1: (1, 7),  # normal
                2: (1, 8),  # hard
                3: (1, 10),  # oni
                4: (1, 10),  # ura
            }
        self.star_ranges = star_ranges

        # Create calibrators per difficulty
        if method == "spline":
            self.calibrators = nn.ModuleList(
                [
                    MonotonicSpline(
                        n_knots=8,
                        input_range=(0, 15),
                        output_range=star_ranges.get(i, (1, 10)),
                    )
                    for i in range(n_difficulties)
                ]
            )
        else:
            self.calibrators = nn.ModuleList(
                [MonotonicMLP(d_hidden=32, n_layers=3) for i in range(n_difficulties)]
            )

            # Add scaling parameters for MLP
            self.scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(
                            float(
                                star_ranges.get(i, (1, 10))[1]
                                - star_ranges.get(i, (1, 10))[0]
                            )
                        )
                    )
                    for i in range(n_difficulties)
                ]
            )
            self.offsets = nn.ParameterList(
                [
                    nn.Parameter(torch.tensor(float(star_ranges.get(i, (1, 10))[0])))
                    for i in range(n_difficulties)
                ]
            )

    def forward(
        self,
        raw_score: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map raw scores to star ratings based on difficulty.

        Args:
            raw_score: [batch] raw difficulty scores
            difficulty: [batch] difficulty class indices

        Returns:
            star_rating: [batch] calibrated star ratings (can be < min or > max)
        """
        batch_size = raw_score.size(0)
        star_ratings = torch.zeros_like(raw_score)

        # Process each difficulty class
        for diff_idx in range(self.n_difficulties):
            mask = difficulty == diff_idx
            if mask.any():
                calibrator = self.calibrators[diff_idx]

                if self.method == "spline":
                    star_ratings[mask] = calibrator(raw_score[mask])
                else:
                    # MLP with scaling
                    normalized = calibrator(raw_score[mask])
                    star_ratings[mask] = (
                        normalized * self.scales[diff_idx] + self.offsets[diff_idx]
                    )

        return star_ratings

    def forward_all(
        self,
        raw_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute star ratings for all difficulties at once.

        Args:
            raw_score: [batch] raw scores

        Returns:
            star_ratings: [batch, n_difficulties] star per difficulty
        """
        batch_size = raw_score.size(0)
        all_stars = []

        for diff_idx in range(self.n_difficulties):
            calibrator = self.calibrators[diff_idx]

            if self.method == "spline":
                stars = calibrator(raw_score)
            else:
                normalized = calibrator(raw_score)
                stars = normalized * self.scales[diff_idx] + self.offsets[diff_idx]

            all_stars.append(stars)

        return torch.stack(all_stars, dim=-1)

    def clip_to_display(
        self,
        star_rating: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip star ratings to display range for UI.

        Args:
            star_rating: [batch] raw star ratings (can be outside range)
            difficulty: [batch] difficulty indices

        Returns:
            display_star: [batch] clipped to valid range per difficulty
        """
        display_star = star_rating.clone()

        for diff_idx in range(self.n_difficulties):
            mask = difficulty == diff_idx
            if mask.any():
                min_star, max_star = self.star_ranges[diff_idx]
                display_star[mask] = display_star[mask].clamp(min_star, max_star)

        return display_star
