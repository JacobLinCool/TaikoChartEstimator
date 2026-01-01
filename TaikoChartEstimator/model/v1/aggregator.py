"""
MIL Bag Aggregator for Taiko Chart Estimation

Implements Multiple Instance Learning aggregation with:
- Three-way pooling (mean, top-k, attention)
- Multi-branch attention (ACMIL-inspired)
- Stochastic top-k masking to prevent attention collapse
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBranch(nn.Module):
    """Single attention branch for multi-branch attention."""

    def __init__(self, d_instance: int, d_hidden: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_instance, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instances: [batch, n_instances, d_instance]
            mask: [batch, n_instances], 1 for valid, 0 for padding

        Returns:
            pooled: [batch, d_instance]
            attention_weights: [batch, n_instances]
        """
        # Compute attention scores
        scores = self.attention(instances).squeeze(-1)  # [batch, n_instances]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Handle all-masked case
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, 0.0)

        # Weighted sum
        pooled = (instances * attn_weights.unsqueeze(-1)).sum(dim=1)

        return pooled, attn_weights


class MILAggregator(nn.Module):
    """
    Multiple Instance Learning aggregator with ACMIL-inspired design.

    Combines three complementary pooling strategies:
    1. Mean pooling: Captures overall difficulty/stamina
    2. Top-K pooling: Captures peak difficulty segments
    3. Multi-branch attention: Learns multiple discriminative patterns

    Features stochastic top-k masking during training to prevent
    the model from relying on only a few "hardest" instances.
    """

    def __init__(
        self,
        d_instance: int = 256,
        n_branches: int = 3,
        top_k_ratio: float = 0.1,
        stochastic_mask_prob: float = 0.3,
        dropout: float = 0.1,
    ):
        """
        Initialize MIL aggregator.

        Args:
            d_instance: Dimension of instance embeddings
            n_branches: Number of attention branches
            top_k_ratio: Fraction of instances for top-k pooling
            stochastic_mask_prob: Probability of masking top instances during training
            dropout: Dropout rate
        """
        super().__init__()

        self.d_instance = d_instance
        self.n_branches = n_branches
        self.top_k_ratio = top_k_ratio
        self.stochastic_mask_prob = stochastic_mask_prob

        # Top-K scoring network
        self.topk_scorer = nn.Sequential(
            nn.Linear(d_instance, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Multi-branch attention
        self.attention_branches = nn.ModuleList(
            [AttentionBranch(d_instance, d_hidden=64) for _ in range(n_branches)]
        )

        # Fusion layer: combines mean (1) + topk (1) + branches (n_branches) = 2 + n_branches
        n_pooled = 2 + n_branches
        self.fusion = nn.Sequential(
            nn.Linear(d_instance * n_pooled, d_instance * 2),
            nn.LayerNorm(d_instance * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_instance * 2, d_instance * 2),
        )

        self.output_dim = d_instance * 2

    def _mean_pool(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean pooling over instances."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            pooled = (instances * mask_expanded).sum(dim=1)
            pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = instances.mean(dim=1)
        return pooled

    def _topk_pool(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K pooling based on learned scores.

        Returns:
            pooled: [batch, d_instance]
            topk_mask: [batch, n_instances] binary mask of selected instances
        """
        batch_size, n_instances, _ = instances.shape

        # Compute scores
        scores = self.topk_scorer(instances).squeeze(-1)  # [batch, n_instances]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Determine k
        if mask is not None:
            valid_counts = mask.sum(dim=1)  # [batch]
            k = (valid_counts * self.top_k_ratio).clamp(min=1).long()
            max_k = k.max().item()
        else:
            k = max(1, int(n_instances * self.top_k_ratio))
            max_k = k

        # Get top-k indices
        _, topk_indices = scores.topk(max_k, dim=1)  # [batch, max_k]

        # Create topk mask
        topk_mask = torch.zeros_like(mask if mask is not None else scores)
        topk_mask.scatter_(1, topk_indices, 1.0)

        # Pool top-k instances
        if mask is not None:
            combined_mask = topk_mask * mask
        else:
            combined_mask = topk_mask

        mask_expanded = combined_mask.unsqueeze(-1)
        pooled = (instances * mask_expanded).sum(dim=1)
        pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1)

        return pooled, topk_mask

    def _stochastic_topk_mask(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create stochastic mask that randomly drops top instances.

        This prevents attention collapse by forcing the model to
        learn from non-peak instances during training.
        """
        if not self.training:
            return mask

        batch_size, n_instances, _ = instances.shape

        # Get top-k scores
        with torch.no_grad():
            scores = self.topk_scorer(instances).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            k = max(1, int(n_instances * self.top_k_ratio))
            _, topk_indices = scores.topk(k, dim=1)

        # Create mask that drops top instances with some probability
        drop_mask = torch.ones_like(mask if mask is not None else scores)

        # For each batch, randomly decide whether to drop top instances
        drop_decision = (
            torch.rand(batch_size, device=instances.device) < self.stochastic_mask_prob
        )

        for i in range(batch_size):
            if drop_decision[i]:
                drop_mask[i, topk_indices[i]] = 0.0

        if mask is not None:
            return mask * drop_mask
        return drop_mask

    def forward(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate instance embeddings to bag embedding.

        Args:
            instances: [batch, n_instances, d_instance]
            mask: [batch, n_instances], 1 for valid, 0 for padding
            return_attention: Whether to return attention weights for analysis

        Returns:
            bag_embedding: [batch, output_dim]
            attention_info: Dict with attention weights and metrics
        """
        # Apply stochastic top-k masking during training
        if self.training:
            stoch_mask = self._stochastic_topk_mask(instances, mask)
        else:
            stoch_mask = mask

        # 1. Mean pooling (stamina/overall representation)
        mean_pooled = self._mean_pool(instances, mask)

        # 2. Top-K pooling (peak difficulty)
        topk_pooled, topk_mask = self._topk_pool(instances, mask)

        # 3. Multi-branch attention pooling
        branch_outputs = []
        branch_attns = []

        for branch in self.attention_branches:
            pooled, attn = branch(instances, stoch_mask)
            branch_outputs.append(pooled)
            branch_attns.append(attn)

        # Concatenate all pooled representations
        all_pooled = [mean_pooled, topk_pooled] + branch_outputs
        concatenated = torch.cat(
            all_pooled, dim=-1
        )  # [batch, d_instance * (2 + n_branches)]

        # Fuse
        bag_embedding = self.fusion(concatenated)

        # Compute attention health metrics
        attention_info = {}
        if return_attention:
            # Stack all attention weights
            all_attn = torch.stack(
                branch_attns, dim=1
            )  # [batch, n_branches, n_instances]

            # Average attention across branches
            avg_attn = all_attn.mean(dim=1)  # [batch, n_instances]

            # Attention entropy (higher = more distributed)
            entropy = -(avg_attn * (avg_attn + 1e-8).log()).sum(dim=-1)

            # Effective number of instances (inverse of concentration)
            effective_n = 1.0 / (avg_attn**2).sum(dim=-1)

            # Top-5% mass
            k = max(1, int(instances.size(1) * 0.05))
            top5_mass = avg_attn.topk(k, dim=-1).values.sum(dim=-1)

            attention_info = {
                "branch_attentions": all_attn,  # [batch, n_branches, n_instances]
                "average_attention": avg_attn,  # [batch, n_instances]
                "topk_mask": topk_mask,  # [batch, n_instances]
                "entropy": entropy,  # [batch]
                "effective_n": effective_n,  # [batch]
                "top5_mass": top5_mass,  # [batch]
            }

        return bag_embedding, attention_info


class GatedMILAggregator(nn.Module):
    """
    Alternative MIL aggregator using gated attention.

    Allows instance embeddings to modulate attention via gating,
    which can capture more nuanced importance patterns.
    """

    def __init__(
        self,
        d_instance: int = 256,
        d_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention_v = nn.Sequential(
            nn.Linear(d_instance, d_hidden),
            nn.Tanh(),
        )

        self.attention_u = nn.Sequential(
            nn.Linear(d_instance, d_hidden),
            nn.Sigmoid(),
        )

        self.attention_w = nn.Linear(d_hidden, 1)

        self.output_proj = nn.Sequential(
            nn.Linear(d_instance, d_instance * 2),
            nn.LayerNorm(d_instance * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = d_instance * 2

    def forward(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            instances: [batch, n_instances, d_instance]
            mask: [batch, n_instances]

        Returns:
            bag_embedding: [batch, output_dim]
            attention_info: Dict with attention weights
        """
        # Gated attention
        v = self.attention_v(instances)  # [batch, n_instances, d_hidden]
        u = self.attention_u(instances)  # [batch, n_instances, d_hidden]

        scores = self.attention_w(v * u).squeeze(-1)  # [batch, n_instances]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, 0.0)

        # Weighted sum
        pooled = (instances * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Project to output
        bag_embedding = self.output_proj(pooled)

        attention_info = {"attention": attn_weights} if return_attention else {}

        return bag_embedding, attention_info
