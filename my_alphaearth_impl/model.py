"""A compact AlphaEarth-style model with clear comments.

This module intentionally stays small and readable:
- Space branch: spatial feature extraction per time step.
- Time branch: attention-like weighting over temporal embeddings.
- Precision branch: local residual refinement.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Lightweight temporal pooling.

    Input:  x [B, T, D]
    Output: y [B, D]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention score per time step.
        weights = self.score(x)  # [B, T, 1]
        weights = torch.softmax(weights, dim=1)
        # Weighted sum across time.
        return (x * weights).sum(dim=1)


class SimpleSTPModel(nn.Module):
    """Simplified STP-inspired encoder-decoder.

    Expected input shape: [B, T, C, H, W]
    Returns reconstructed image: [B, C, H, W]
    """

    def __init__(self, in_channels: int = 6, embed_dim: int = 128):
        super().__init__()

        # Space operator: extract per-frame spatial features.
        self.space_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project spatial feature to shared embedding.
        self.to_embed = nn.Linear(64, embed_dim)

        # Time operator: aggregate all temporal embeddings.
        self.time_pool = TemporalAttentionPool(embed_dim)

        # Precision operator: decode + refine local detail.
        self.decoder_seed = nn.Linear(embed_dim, 64)
        self.precision_decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape

        # 1) Space branch per time step.
        frames = x.view(b * t, c, h, w)
        spatial_feat = self.space_encoder(frames).flatten(1)  # [B*T, 64]

        # 2) Map to temporal token sequence [B, T, D].
        tokens = self.to_embed(spatial_feat).view(b, t, -1)

        # 3) Time branch aggregation -> [B, D].
        global_embed = self.time_pool(tokens)

        # 4) Decode a full-resolution reconstruction.
        seed = self.decoder_seed(global_embed).view(b, 64, 1, 1)
        seed = seed.expand(-1, -1, h, w)
        out = self.precision_decoder(seed)
        return out
