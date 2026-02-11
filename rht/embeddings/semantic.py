"""Semantic encoder phi_sem for PentE."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    """
    Encodes raw numeric attributes into a fixed-dim semantic representation.

    This is a simple MLP; in production it could incorporate pre-trained
    embeddings (e.g. Word2Vec on textual descriptions — Section 6.2, step 1).
    """

    def __init__(self, out_dim: int = 128, max_in_dim: int = 2048):
        super().__init__()
        self.out_dim = out_dim
        # Lazy linear adapts to actual input dim on first forward
        self.net = nn.Sequential(
            nn.LazyLinear(out_dim * 2),
            nn.GELU(),
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
