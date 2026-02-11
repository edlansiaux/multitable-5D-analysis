"""
PentE: Pentadimensional Embedding.

Implements Definition 5 (Section 5.3) and Appendix A.2:

    z_e = phi_sem(e) || phi_rel(e) || phi_temp(e) || phi_cat(e) || phi_vol(e)

with three regularisation constraints:
  1. Relational preservation
  2. Temporal continuity
  3. Categorical similarity
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rht.embeddings.semantic import SemanticEncoder
from rht.embeddings.categorical import HierarchicalCategoricalEncoder
from rht.embeddings.volume import VolumeNormalizer
from rht.modules.temporal import MultiScaleTemporalEmbedding


class RelationalGNN(nn.Module):
    """
    Lightweight GNN encoder for the relational component phi_rel.

    Uses a simple 2-layer message-passing scheme over the adjacency.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor | None = None
    ) -> torch.Tensor:
        if adj is not None:
            # Simple mean-aggregation message passing
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            x = adj @ x / deg
        x = torch.relu(self.lin1(x))
        if adj is not None:
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            x = adj @ x / deg
        x = self.lin2(x)
        return x


class PentEEmbedding(nn.Module):
    """
    Full PentE embedding module (Appendix A.2, Listing 3).

    Combines five dimension-specific encoders and concatenates their outputs.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.semantic_encoder = SemanticEncoder(config.semantic_dim)
        self.relational_encoder = RelationalGNN(
            config.semantic_dim, config.relational_dim
        )
        self.temporal_encoder = MultiScaleTemporalEmbedding(
            out_dim=config.temporal_dim
        )
        self.categorical_encoder = HierarchicalCategoricalEncoder(
            embed_dim=config.categorical_dim,
        )
        self.volume_normalizer = VolumeNormalizer(config.volume_dim)

    def forward(
        self,
        node_features: dict[str, torch.Tensor] | torch.Tensor,
        timestamps: torch.Tensor | None = None,
        categories: dict[str, torch.Tensor] | torch.Tensor | None = None,
        hypergraph: Any | None = None,
    ) -> torch.Tensor:
        """
        Compute PentE embeddings (Eq. 5).

        If node_features is a dict of per-table tensors they are concatenated
        along the node axis.  If a plain tensor, used directly.
        """
        # --- Flatten per-table features if needed ---
        if isinstance(node_features, dict):
            x = torch.cat(list(node_features.values()), dim=0)
        else:
            x = node_features

        # Dim 1 — Semantic
        sem_emb = self.semantic_encoder(x)

        # Dim 2 — Relational (message passing over adjacency)
        adj = None
        if hypergraph is not None and hasattr(hypergraph, "adjacency_from_incidence"):
            adj = hypergraph.adjacency_from_incidence().to(sem_emb.device)
        rel_emb = self.relational_encoder(sem_emb, adj)

        # Dim 3 — Temporal
        if timestamps is not None:
            temp_emb = self.temporal_encoder(timestamps)
        else:
            temp_emb = torch.zeros(
                x.shape[0],
                self.temporal_encoder.projection.out_features,
                device=x.device,
            )

        # Dim 4 — Categorical
        if categories is not None:
            if isinstance(categories, dict):
                cat_input = torch.cat(list(categories.values()), dim=0)
            else:
                cat_input = categories
            cat_emb = self.categorical_encoder(cat_input)
        else:
            cat_emb = torch.zeros(
                x.shape[0], self.categorical_encoder.embed_dim, device=x.device
            )

        # Dim 5 — Volume
        vol_emb = self.volume_normalizer(x)

        # Concatenation (Eq. 5)
        pente_emb = torch.cat(
            [sem_emb, rel_emb, temp_emb, cat_emb, vol_emb], dim=-1
        )
        return pente_emb
