"""
Eight-Step Multi-Table Analysis Pipeline.

Orchestrates the four RHT modules through the eight operational steps
described in Section 5 and mapped in Figure 1.

  S1 -> Meta-Profiling         (pipeline.profiling)
  S2 -> Hypergraph Construction (Module 1)
  S3 -> PentE Embedding         (Modules 1+2+3)
  S4 -> Contrastive Learning    (Modules 2+3)
  S5 -> Dynamic Rewiring        (pipeline.rewiring)
  S6 -> Causal Inference         (pipeline.causal)
  S7 -> Federated Learning       (pipeline.federated)
  S8 -> Deployment & Monitoring  (Module 4)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from rht.model import RelationalHypergraphTransformer, RHTConfig
from rht.modules.hypergraph import AdaptiveHypergraphConstructor, RelationalHypergraph
from rht.losses.contrastive import RelationalTemporalContrastiveLoss
from rht.losses.relational import RelationalDiscoveryLoss
from pipeline.profiling import MetaProfiler
from pipeline.rewiring import DynamicGraphRewirer

logger = logging.getLogger(__name__)


class MultiTablePipeline:
    """
    End-to-end pipeline implementing the eight steps of Section 5.

    Parameters
    ----------
    tables : dict[str, DataFrame]
        Input relational tables keyed by name.
    relations : list of (table1, key1, table2, key2)
        Explicit foreign-key relations.
    config : RHTConfig, optional
        Model configuration.
    """

    def __init__(
        self,
        tables: dict[str, Any],
        relations: list[tuple[str, str, str, str]] | None = None,
        config: RHTConfig | None = None,
    ):
        self.tables = tables
        self.explicit_relations = relations or []
        self.config = config or RHTConfig()

        self.profiler = MetaProfiler()
        self.hypergraph_constructor = AdaptiveHypergraphConstructor()
        self.rewirer = DynamicGraphRewirer()
        self.model = RelationalHypergraphTransformer(self.config)

        # State populated during the pipeline
        self.profile: dict | None = None
        self.hypergraph: RelationalHypergraph | None = None
        self.adjacency: torch.Tensor | None = None
        self._pente_embeddings: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Step 1: Predictive Meta-Profiling (Section 5.1)
    # ------------------------------------------------------------------
    def step1_meta_profiling(self):
        logger.info("Step 1 — Meta-Profiling")
        self.profile = self.profiler.profile(self.tables)
        logger.info(
            "Dimensional profile: %s", self.profile.get("dimensional_profile")
        )
        return self.profile

    # ------------------------------------------------------------------
    # Step 2: Automated Hypergraph Construction (Section 5.2)
    # ------------------------------------------------------------------
    def step2_hypergraph_construction(self):
        logger.info("Step 2 — Hypergraph Construction")
        self.hypergraph = self.hypergraph_constructor.construct_hypergraph(
            self.tables, self.explicit_relations or None
        )
        self.adjacency = self.hypergraph.adjacency_from_incidence()
        logger.info(
            "Hypergraph: %d nodes, %d hyperedges",
            self.hypergraph.num_nodes,
            self.hypergraph.num_edges,
        )
        return self.hypergraph

    # ------------------------------------------------------------------
    # Step 3: PentE Embedding (Section 5.3)
    # ------------------------------------------------------------------
    def step3_pente_embedding(self, timestamps=None, categories=None):
        logger.info("Step 3 — PentE Embedding")
        import pandas as pd

        # Build per-table numeric feature tensors
        node_features = {}
        for tname, df in self.tables.items():
            if isinstance(df, pd.DataFrame):
                numeric = df.select_dtypes(include="number")
                node_features[tname] = torch.tensor(
                    numeric.values, dtype=torch.float32
                )
        self._pente_embeddings = self.model.pente(
            node_features=node_features,
            timestamps=timestamps,
            categories=categories,
            hypergraph=self.hypergraph,
        )
        return self._pente_embeddings

    # ------------------------------------------------------------------
    # Step 4: Relational Contrastive Learning (Section 5.4)
    # ------------------------------------------------------------------
    def step4_contrastive_learning(
        self,
        epochs: int = 50,
        lr: float | None = None,
        timestamps: torch.Tensor | None = None,
    ):
        logger.info(
            "Step 4 — Relational Contrastive Learning (%d epochs)", epochs
        )
        crt_loss = RelationalTemporalContrastiveLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr or self.config.learning_rate
        )

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(
                node_features=self._build_node_features(),
                timestamps=timestamps,
                adjacency=self.adjacency,
                hypergraph=self.hypergraph,
            )
            loss = crt_loss(out["embeddings"], self.adjacency, timestamps)
            loss = loss + out["discovery_loss"]
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "  epoch %d / %d  loss=%.4f",
                    epoch + 1,
                    epochs,
                    loss.item(),
                )

    # ------------------------------------------------------------------
    # Step 5: Dynamic Graph Rewiring (Section 5.5)
    # ------------------------------------------------------------------
    def step5_dynamic_rewiring(self):
        logger.info("Step 5 — Dynamic Graph Rewiring")
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                node_features=self._build_node_features(),
                adjacency=self.adjacency,
                hypergraph=self.hypergraph,
            )
        self.adjacency = self.rewirer.rewire(
            self.adjacency, out["attention_weights"]
        )

    # ------------------------------------------------------------------
    # Step 6: Relational Causal Inference (Section 5.6) — stub
    # ------------------------------------------------------------------
    def step6_causal_inference(self):
        logger.info("Step 6 — Causal Inference (placeholder)")
        # Full implementation: Double ML, Granger causality on graphs, do-calculus
        pass

    # ------------------------------------------------------------------
    # Step 7: Federated Learning (Section 5.7) — stub
    # ------------------------------------------------------------------
    def step7_federated_learning(self):
        logger.info("Step 7 — Federated Learning (placeholder)")
        # Full implementation: local training on schema fragments + secure aggregation
        pass

    # ------------------------------------------------------------------
    # Step 8: Operationalization & Monitoring (Section 5.8)
    # ------------------------------------------------------------------
    def step8_deployment(self):
        logger.info("Step 8 — Deployment & Monitoring (placeholder)")
        pass

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def run(self, contrastive_epochs: int = 50):
        """Execute all eight steps sequentially."""
        self.step1_meta_profiling()
        self.step2_hypergraph_construction()
        self.step3_pente_embedding()
        self.step4_contrastive_learning(epochs=contrastive_epochs)
        self.step5_dynamic_rewiring()
        self.step6_causal_inference()
        self.step7_federated_learning()
        self.step8_deployment()

    def get_pente_embeddings(self) -> torch.Tensor | None:
        return self._pente_embeddings

    def predict(self, task: str = "default", **kwargs):
        """Run inference with the trained model."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                node_features=self._build_node_features(),
                adjacency=self.adjacency,
                hypergraph=self.hypergraph,
            )
        return out["logits"]

    def _build_node_features(self) -> dict[str, torch.Tensor]:
        import pandas as pd

        node_features = {}
        for tname, df in self.tables.items():
            if isinstance(df, pd.DataFrame):
                numeric = df.select_dtypes(include="number")
                node_features[tname] = torch.tensor(
                    numeric.values, dtype=torch.float32
                )
        return node_features
