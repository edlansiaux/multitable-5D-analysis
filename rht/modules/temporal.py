"""
Module 2: Multi-Scale Temporal Embeddings.

Implements Definition 2 (Multi-Resolution Temporal Embedding) from Section 4.2.2,
inspired by Time2Vec (Kazemi et al., 2019).

    phi_temp(t) = [w0*t, sin(w1*t), sin(w2*t), ..., sin(wk*t)]

where the frequencies {wi} are learned to capture different temporal scales.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Time2Vec embedding (Kazemi et al., 2019) as used in Module 2.

    Parameters
    ----------
    out_dim : int
        Total output dimensionality.  The first component is a learned linear
        term; the remaining ``out_dim - 1`` components are periodic (sin).
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        assert out_dim >= 2, "out_dim must be >= 2 for Time2Vec"
        self.out_dim = out_dim
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        # Periodic components
        self.W = nn.Parameter(torch.randn(out_dim - 1))
        self.B = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : Tensor of shape (*, )
            Timestamps (seconds since epoch, or normalised floats).

        Returns
        -------
        Tensor of shape (*, out_dim)
        """
        t = t.unsqueeze(-1) if t.dim() == 0 or t.shape[-1] != 1 else t
        linear = self.w0 * t + self.b0                          # (*, 1)
        periodic = torch.sin(self.W * t + self.B)               # (*, out_dim-1)
        return torch.cat([linear, periodic], dim=-1)             # (*, out_dim)


class MultiScaleTemporalEmbedding(nn.Module):
    """
    Multi-resolution temporal embedding combining several Time2Vec scales.

    This stacks Time2Vec modules at different pre-initialised frequency ranges
    (seconds, minutes, hours, days, weeks) and projects to a common dimension.
    """

    SCALES = {
        "fine":   1.0,             # seconds-level
        "minute": 1.0 / 60,
        "hour":   1.0 / 3600,
        "day":    1.0 / 86400,
        "week":   1.0 / 604800,
    }

    def __init__(self, per_scale_dim: int = 8, out_dim: int = 32):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for name in self.SCALES:
            self.encoders[name] = Time2Vec(per_scale_dim)
        concat_dim = per_scale_dim * len(self.SCALES)
        self.projection = nn.Linear(concat_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : Tensor of shape (*, )
            Raw timestamps.

        Returns
        -------
        Tensor of shape (*, out_dim)
        """
        parts = []
        for name, scale in self.SCALES.items():
            parts.append(self.encoders[name](t * scale))
        return self.projection(torch.cat(parts, dim=-1))


class TemporalAttentionAlignment(nn.Module):
    """
    Cross-table temporal attention for aligning irregular time series
    (Section 4.2.2, last paragraph).

    Given query timestamps from one table and key timestamps from another,
    computes soft alignment weights.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.temporal_emb = MultiScaleTemporalEmbedding(out_dim=embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
        key_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query_times : (B, Tq)
        key_times   : (B, Tk)
        key_values  : (B, Tk, D)

        Returns
        -------
        Aligned values of shape (B, Tq, D).
        """
        q_emb = self.temporal_emb(query_times)   # (B, Tq, d)
        k_emb = self.temporal_emb(key_times)     # (B, Tk, d)
        aligned, _ = self.attn(q_emb, k_emb, key_values)
        return aligned
