"""
Relational Hypergraph Transformer (RHT) — main model.

Implements the unified architecture described in Section 4 of the paper,
combining four computational modules:
  - Module 1: Hypergraph Construction      (Layer 1 — Semantic Abstraction)
  - Module 2: Multi-Scale Temporal Emb.    (Layer 2 — Relational Engineering)
  - Module 3: High-Cardinality Attention   (Layer 3 — Trans-Tabular Analytics)
  - Module 4: Differentiable Rel. Discovery(Layer 4 — Operational Orchestration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from rht.modules.hypergraph import AdaptiveHypergraphConstructor, RelationalHypergraph
from rht.modules.attention import HighCardinalityAttentionBlock
from rht.modules.discovery import DifferentiableRelationalDiscovery
from rht.embeddings.pente import PentEEmbedding


@dataclass
class RHTConfig:
    """Configuration for the Relational Hypergraph Transformer."""

    # Embedding dimensions (per component of PentE, cf. Definition 5)
    semantic_dim: int = 128
    relational_dim: int = 64
    temporal_dim: int = 32
    categorical_dim: int = 64
    volume_dim: int = 16

    # Attention (Module 3, Definition 3)
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Temporal (Module 2, Definition 2)
    num_temporal_frequencies: int = 16

    # Discovery (Module 4, Definition 4)
    discovery_alpha: float = 1.0
    discovery_beta: float = 0.1
    discovery_gamma: float = 0.05

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_message_passing_layers: int = 3

    # Rare-category memory bank (Section 4.2.3)
    memory_bank_size: int = 1024
    few_shot_k: int = 5

    @property
    def pente_dim(self) -> int:
        """Total PentE embedding dimensionality (Eq. 5)."""
        return (
            self.semantic_dim
            + self.relational_dim
            + self.temporal_dim
            + self.categorical_dim
            + self.volume_dim
        )


class RelationalHypergraphTransformer(nn.Module):
    """
    End-to-end Relational Hypergraph Transformer.

    This model implements the four-module architecture from Figure 3 of the paper
    and exposes a forward pass that chains:
        Hypergraph -> Temporal Embedding -> Sparse Attention -> Discovery.
    """

    def __init__(self, config: RHTConfig | None = None):
        super().__init__()
        self.config = config or RHTConfig()

        # --- Module 1: Hypergraph Construction (Section 4.2.1) ---
        self.hypergraph_constructor = AdaptiveHypergraphConstructor()

        # --- PentE Embedding (Definition 5) ---
        self.pente = PentEEmbedding(self.config)

        # --- Module 2 + 3: Stacked message-passing + attention layers ---
        self.layers = nn.ModuleList()
        for _ in range(self.config.num_message_passing_layers):
            self.layers.append(
                HighCardinalityAttentionBlock(
                    embed_dim=self.config.pente_dim,
                    num_heads=self.config.num_attention_heads,
                    dropout=self.config.attention_dropout,
                )
            )

        # --- Module 4: Differentiable Relational Discovery (Section 4.2.4) ---
        self.discovery = DifferentiableRelationalDiscovery(
            embed_dim=self.config.pente_dim,
            alpha=self.config.discovery_alpha,
            beta=self.config.discovery_beta,
            gamma=self.config.discovery_gamma,
        )

        # Projection head for downstream tasks
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.config.pente_dim),
            nn.Linear(self.config.pente_dim, self.config.pente_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.pente_dim // 2, 1),
        )

    def forward(
        self,
        node_features: dict[str, torch.Tensor],
        timestamps: torch.Tensor | None = None,
        categories: dict[str, torch.Tensor] | None = None,
        hypergraph: RelationalHypergraph | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through the four modules.

        Parameters
        ----------
        node_features : dict[str, Tensor]
            Per-table feature tensors keyed by table name.
        timestamps : Tensor, optional
            Timestamps for temporal embedding (Module 2).
        categories : dict[str, Tensor], optional
            Categorical variable indices per table.
        hypergraph : RelationalHypergraph, optional
            Pre-built hypergraph (if None, built on the fly via Module 1).
        adjacency : Tensor, optional
            Adjacency / incidence matrix for sparse attention masking.

        Returns
        -------
        dict with keys:
            "embeddings" — PentE embeddings (N x D)
            "logits"     — task output
            "discovery_loss" — relational discovery auxiliary loss
            "attention_weights" — last-layer attention weights
        """
        # --- Step 3: PentE Embedding ---
        z = self.pente(
            node_features=node_features,
            timestamps=timestamps,
            categories=categories,
            hypergraph=hypergraph,
        )

        # --- Steps 4-5: Attention layers (message passing + rewiring) ---
        attn_weights = None
        for layer in self.layers:
            z, attn_weights = layer(z, adjacency=adjacency)

        # --- Module 4: Discovery loss ---
        discovery_loss = self.discovery(z, adjacency)

        # --- Task head ---
        logits = self.output_head(z)

        return {
            "embeddings": z,
            "logits": logits,
            "discovery_loss": discovery_loss,
            "attention_weights": attn_weights,
        }
