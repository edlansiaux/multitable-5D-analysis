"""
Hierarchical High-Cardinality Categorical Encoder (Section 6.2).

Three-level encoding:
  1. Initial embedding (pre-trained or learned)
  2. Hierarchical clustering -> cluster embedding
  3. Composite: e_c = e_code || e_cluster || e_parent   (Eq. 7)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalCategoricalEncoder(nn.Module):
    """
    Encodes high-cardinality categorical variables via the three-level
    hierarchy described in Section 6.2.

    Parameters
    ----------
    embed_dim : int
        Final embedding dimensionality per category.
    num_codes : int
        Maximum vocabulary size (e.g. 15 000 for ICD-10 codes).
    num_clusters : int
        Number of intermediate clusters.
    num_parents : int
        Number of top-level parent groups.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_codes: int = 15_000,
        num_clusters: int = 500,
        num_parents: int = 50,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        sub = embed_dim // 3
        remainder = embed_dim - 2 * sub

        # Level 1 — code embedding
        self.code_emb = nn.Embedding(num_codes, sub, padding_idx=0)

        # Level 2 — cluster embedding
        self.cluster_emb = nn.Embedding(num_clusters, sub, padding_idx=0)

        # Level 3 — parent embedding
        self.parent_emb = nn.Embedding(num_parents, remainder, padding_idx=0)

        # Learned mapping: code -> cluster, code -> parent
        # (In practice, these would be populated from an ICD-10 hierarchy file.)
        self.register_buffer(
            "code_to_cluster", torch.zeros(num_codes, dtype=torch.long)
        )
        self.register_buffer(
            "code_to_parent", torch.zeros(num_codes, dtype=torch.long)
        )

    def set_hierarchy(
        self,
        code_to_cluster: torch.LongTensor,
        code_to_parent: torch.LongTensor,
    ):
        """Load the cluster / parent mappings (e.g. from ICD-10 chapter structure)."""
        self.code_to_cluster.copy_(code_to_cluster)
        self.code_to_parent.copy_(code_to_parent)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        codes : LongTensor of shape (*, )
            Category indices.

        Returns
        -------
        Composite embedding of shape (*, embed_dim)  — Eq. 7.
        """
        codes = codes.long()
        e_code = self.code_emb(codes)

        clusters = self.code_to_cluster[codes]
        e_cluster = self.cluster_emb(clusters)

        parents = self.code_to_parent[codes]
        e_parent = self.parent_emb(parents)

        return torch.cat([e_code, e_cluster, e_parent], dim=-1)
