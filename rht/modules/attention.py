"""
Module 3: High-Cardinality Sparse Relational Attention.

Implements Definition 3 (Sparse Relational Attention) and Listing 1 from
Section 6.1.  Complexity is reduced from O(n^2) to O(n*k) where k is the
average degree in the relational graph.

    Attention(Q, K, V) = softmax( (Q*K^T) / sqrt(dk) + M_mask ) * V

where M_mask[i,j] = 0 if i and j are relationally connected, else -inf.

Also includes a memory bank for few-shot learning on rare categories
(Section 4.2.3, last paragraph).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseRelationalAttention(nn.Module):
    """
    Sparse relational attention (Definition 3, Listing 1).

    Parameters
    ----------
    embed_dim : int
        Model / embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability on attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (B, N, D) or (N, D)
            Node embeddings.
        adjacency : Tensor of shape (N, N), optional
            Binary adjacency matrix.  Entries of 0 are masked to -inf.

        Returns
        -------
        output : Tensor same shape as x
        attn_weights : (B, H, N, N) or (H, N, N)
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        Q = self.W_q(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = self.W_k(x).view(B, N, H, d).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, d).transpose(1, 2)

        # Scaled dot-product scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)  # (B, H, N, N)

        # --- Sparse masking based on relational structure (Eq. 2-3) ---
        if adjacency is not None:
            mask = (adjacency == 0).float() * (-1e9)
            # Broadcast over batch and heads
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)                       # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)      # (B, N, D)
        out = self.out_proj(out)

        if squeeze:
            out = out.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return out, attn_weights


class RareCategoryMemoryBank(nn.Module):
    """
    Memory bank for few-shot learning on rare categories (Section 4.2.3).

    Stores prototypes (mean embeddings) for each known category and enables
    nearest-prototype classification for categories with very few examples.
    """

    def __init__(self, embed_dim: int, max_categories: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_categories = max_categories
        # Prototype storage (not a parameter — updated via EMA)
        self.register_buffer("prototypes", torch.zeros(max_categories, embed_dim))
        self.register_buffer("counts", torch.zeros(max_categories, dtype=torch.long))

    @torch.no_grad()
    def update(
        self, embeddings: torch.Tensor, labels: torch.LongTensor, momentum: float = 0.9
    ):
        """Update prototypes with exponential moving average."""
        for lbl in labels.unique():
            idx = lbl.item()
            if idx >= self.max_categories:
                continue
            mask = labels == lbl
            mean_emb = embeddings[mask].mean(0)
            if self.counts[idx] == 0:
                self.prototypes[idx] = mean_emb
            else:
                self.prototypes[idx] = (
                    momentum * self.prototypes[idx] + (1 - momentum) * mean_emb
                )
            self.counts[idx] += mask.sum()

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Nearest-prototype classification."""
        active = self.counts > 0
        protos = self.prototypes[active]
        # Cosine similarity
        sim = F.cosine_similarity(
            embeddings.unsqueeze(1), protos.unsqueeze(0), dim=-1
        )
        return sim.argmax(dim=-1)


class HighCardinalityAttentionBlock(nn.Module):
    """
    Full attention block used inside the RHT: sparse relational attention
    + feed-forward + residual + layer norm (standard Transformer block layout
    adapted with the relational mask from Module 3).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.attn = SparseRelationalAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_w = self.attn(x_norm, adjacency=adjacency)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_w
