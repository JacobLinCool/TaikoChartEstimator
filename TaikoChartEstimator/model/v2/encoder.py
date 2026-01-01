"""
Instance Encoder for Taiko Chart MIL - Version 2

Encodes a sequence of event tokens into a fixed-size vector representation.
Uses Transformer encoder for capturing rhythm patterns and dependencies.

V2 Changes:
- Updated for new feature set: beat_pos, duration, bpm, scroll, local_density
- Removed gogo, added local_density
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContinuousFeatureEncoder(nn.Module):
    """
    Encodes continuous features to d_model dimension.
    Uses learned linear projections with optional normalization.

    V2 Features (5 total):
    - beat_position (within measure, 0-1)
    - duration (for long notes)
    - bpm (normalized)
    - scroll (normalized)
    - local_density (notes/sec, normalized)
    """

    def __init__(
        self,
        n_continuous: int = 5,  # beat_pos, duration, bpm, scroll, local_density
        d_model: int = 256,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.projection = nn.Linear(n_continuous, d_model)
        self.layernorm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Continuous features [batch, seq_len, n_continuous]
        """
        return self.layernorm(self.projection(x))


class InstanceEncoder(nn.Module):
    """
    Encodes a sequence of event tokens to a fixed-size vector.

    Input: Token sequence [batch, seq_len, 6]
        - Column 0: note_type (discrete, 0-9)
        - Column 1: beat_position (continuous, 0-1)
        - Column 2: duration (continuous, normalized)
        - Column 3: bpm (continuous, normalized)
        - Column 4: scroll (continuous, normalized)
        - Column 5: local_density (continuous, normalized)

    Output: Instance embedding [batch, d_model]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        n_note_types: int = 10,  # 9 types + padding
        max_seq_len: int = 128,
        pooling: str = "cls",  # "cls", "mean", or "max"
    ):
        """
        Initialize instance encoder.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_feedforward: Feedforward dimension
            dropout: Dropout rate
            n_note_types: Number of note type categories
            max_seq_len: Maximum sequence length
            pooling: Pooling strategy for sequence to vector
        """
        super().__init__()

        self.d_model = d_model
        self.pooling = pooling

        # Discrete feature embedding (note type)
        self.type_embedding = nn.Embedding(n_note_types, d_model, padding_idx=9)

        # Continuous feature encoder (5 features in v2)
        self.continuous_encoder = ContinuousFeatureEncoder(
            n_continuous=5,  # beat_pos, duration, bpm, scroll, local_density
            d_model=d_model,
        )

        # Feature fusion
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # Positional encoding (max_len+1 to accommodate CLS token)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len + 1, dropout)

        # CLS token for pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode token sequence to vector.

        Args:
            tokens: Token tensor [batch, seq_len, 6]
            mask: Attention mask [batch, seq_len], 1 for valid, 0 for padding

        Returns:
            Instance embedding [batch, d_model]
        """
        batch_size, seq_len, _ = tokens.shape

        # Split discrete and continuous features
        note_types = tokens[:, :, 0].long()  # [batch, seq_len]
        continuous_feats = tokens[:, :, 1:]  # [batch, seq_len, 5]

        # Embed discrete features
        type_emb = self.type_embedding(note_types)  # [batch, seq_len, d_model]

        # Encode continuous features
        cont_emb = self.continuous_encoder(
            continuous_feats
        )  # [batch, seq_len, d_model]

        # Fuse embeddings
        fused = self.fusion(torch.cat([type_emb, cont_emb], dim=-1))
        fused = self.fusion_norm(fused)  # [batch, seq_len, d_model]

        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            fused = torch.cat([cls_tokens, fused], dim=1)  # [batch, 1+seq_len, d_model]

            # Extend mask for CLS token
            if mask is not None:
                cls_mask = torch.ones(
                    batch_size, 1, device=mask.device, dtype=mask.dtype
                )
                mask = torch.cat([cls_mask, mask], dim=1)

        # Add positional encoding
        fused = self.pos_encoder(fused)

        # Create attention mask for transformer (True = ignore)
        if mask is not None:
            attn_mask = mask == 0  # Invert: 0 -> True (ignore)
        else:
            attn_mask = None

        # Apply transformer
        encoded = self.transformer(fused, src_key_padding_mask=attn_mask)

        # Pool to vector
        if self.pooling == "cls":
            output = encoded[:, 0]  # CLS token
        elif self.pooling == "mean":
            if mask is not None:
                # Masked mean (exclude padding)
                mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
                output = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(
                    dim=1
                ).clamp(min=1)
            else:
                output = encoded.mean(dim=1)
        elif self.pooling == "max":
            if mask is not None:
                # Masked max (set padding to -inf)
                mask_expanded = mask.unsqueeze(-1)
                encoded = encoded.masked_fill(mask_expanded == 0, float("-inf"))
            output = encoded.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return self.output_norm(output)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        """
        residual = self.residual(x)

        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = F.gelu(self.norm2(self.conv2(out)))
        out = self.dropout(out)

        return out + residual


class TCNInstanceEncoder(nn.Module):
    """
    Alternative instance encoder using Temporal Convolutional Network.
    Faster than Transformer with stronger local inductive bias.

    V2: Updated for 5 continuous features (removed gogo, added local_density)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_note_types: int = 10,
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.type_embedding = nn.Embedding(n_note_types, d_model // 2, padding_idx=9)
        self.continuous_proj = nn.Linear(5, d_model // 2)  # 5 continuous features

        # TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList(
            [
                TCNBlock(d_model, d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(n_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len, 6]
            mask: [batch, seq_len]

        Returns:
            [batch, d_model]
        """
        # Embed inputs
        note_types = tokens[:, :, 0].long()
        continuous = tokens[:, :, 1:]

        type_emb = self.type_embedding(note_types)
        cont_emb = self.continuous_proj(continuous)

        x = torch.cat([type_emb, cont_emb], dim=-1)  # [batch, seq_len, d_model]

        # Convert to channels-first for conv
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]

        # Apply TCN layers
        for layer in self.tcn_layers:
            x = layer(x)

        # Global average pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # [batch, 1, seq_len]
            x = (x * mask_expanded).sum(dim=-1) / mask_expanded.sum(dim=-1).clamp(min=1)
        else:
            x = x.mean(dim=-1)

        return self.output_norm(x)
