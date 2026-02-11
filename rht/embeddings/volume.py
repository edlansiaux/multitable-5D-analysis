"""Volume normalizer phi_vol for PentE (Dimension 1 — massive volume)."""

from __future__ import annotations

import torch
import torch.nn as nn


class VolumeNormalizer(nn.Module):
    """
    Encodes volumetric statistics (table sizes, record counts) into a
    fixed-dim embedding.  This helps the model adapt its representations
    to the scale of each data source (Dimension 1).
    """

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use simple statistics of the feature vector as a volume proxy
        stats = torch.stack(
            [x.mean(dim=-1), x.std(dim=-1), x.norm(dim=-1)], dim=-1
        )
        return self.net(stats)
