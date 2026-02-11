"""
Module 1: Adaptive Hypergraph Construction.

Implements Definition 1 (Relational Hypergraph) and Algorithm 1 from Section 4.2.1
as well as the detailed pseudo-code from Appendix A.1.

A relational hypergraph  H = (V, E, W, Phi)  where:
  - V: set of nodes (entities / records)
  - E <= 2^V: set of hyperedges (n-ary relationships)
  - W: E -> R+ (edge weights)
  - Phi: V -> R^d (node feature vectors)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RelationalHypergraph:
    """
    Relational hypergraph as per Definition 1.

    Attributes
    ----------
    nodes : list[str]
        Node identifiers.
    node_features : dict[str, np.ndarray]
        Phi mapping: node -> feature vector.
    hyperedges : list[set[str]]
        Each hyperedge is a set of node ids.
    weights : list[float]
        W mapping: weight for each hyperedge.
    metadata : dict
        Auxiliary information (relation types, table origins, etc.).
    """

    nodes: list[str] = field(default_factory=list)
    node_features: dict[str, np.ndarray] = field(default_factory=dict)
    hyperedges: list[set[str]] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.hyperedges)

    def incidence_matrix(self) -> torch.Tensor:
        """
        Build a binary incidence matrix  H in {0,1}^{|V| x |E|}.

        H[v, e] = 1  iff node v belongs to hyperedge e.
        """
        node_idx = {n: i for i, n in enumerate(self.nodes)}
        H = torch.zeros(self.num_nodes, self.num_edges)
        for e_idx, edge in enumerate(self.hyperedges):
            for node in edge:
                if node in node_idx:
                    H[node_idx[node], e_idx] = 1.0
        return H

    def adjacency_from_incidence(self) -> torch.Tensor:
        """
        Derive a node-level adjacency matrix from the incidence matrix.

        A = H . W . H^T  (weighted clique expansion of hyperedges).
        """
        H = self.incidence_matrix()
        W = torch.diag(torch.tensor(self.weights, dtype=torch.float32))
        A = H @ W @ H.T
        # Zero out self-loops
        A.fill_diagonal_(0.0)
        return A


# ---------------------------------------------------------------------------
# Relation detection helpers (Appendix A.1, lines 8-28)
# ---------------------------------------------------------------------------

def _detect_foreign_keys(tables: dict[str, Any]) -> list[dict]:
    """Level 1: Explicit relations via foreign-key column name overlap."""
    relations = []
    table_names = list(tables.keys())
    for i, t1 in enumerate(table_names):
        cols1 = set(tables[t1].columns) if hasattr(tables[t1], "columns") else set()
        for t2 in table_names[i + 1:]:
            cols2 = set(tables[t2].columns) if hasattr(tables[t2], "columns") else set()
            shared = cols1 & cols2
            for col in shared:
                relations.append(
                    {"type": "explicit", "table1": t1, "table2": t2, "key": col}
                )
    return relations


def _detect_value_overlaps(
    tables: dict[str, Any], sample_size: int = 10_000, overlap_threshold: float = 0.3
) -> list[dict]:
    """Level 2: Implicit relations via value-set overlap (Section 5.1 sampling)."""
    import pandas as pd

    relations = []
    table_names = list(tables.keys())
    for i, t1 in enumerate(table_names):
        df1 = tables[t1]
        if not isinstance(df1, pd.DataFrame):
            continue
        for t2 in table_names[i + 1:]:
            df2 = tables[t2]
            if not isinstance(df2, pd.DataFrame):
                continue
            for c1 in df1.columns:
                for c2 in df2.columns:
                    if c1 == c2:
                        continue
                    s1 = set(
                        df1[c1]
                        .dropna()
                        .sample(min(sample_size, len(df1)), random_state=42)
                    )
                    s2 = set(
                        df2[c2]
                        .dropna()
                        .sample(min(sample_size, len(df2)), random_state=42)
                    )
                    if len(s1) == 0 or len(s2) == 0:
                        continue
                    overlap = len(s1 & s2) / min(len(s1), len(s2))
                    if overlap >= overlap_threshold:
                        relations.append(
                            {
                                "type": "implicit",
                                "table1": t1,
                                "col1": c1,
                                "table2": t2,
                                "col2": c2,
                                "overlap": overlap,
                            }
                        )
    return relations


def _infer_semantic_relations(tables: dict[str, Any]) -> list[dict]:
    """Level 3: Semantic relations via column-name similarity (placeholder)."""
    # Full implementation would use pre-trained NLP embeddings on column names
    # and descriptions — left as a stub for now.
    return []


def _detect_temporal_patterns(tables: dict[str, Any]) -> list[dict]:
    """Level 4: Temporal relations — detect datetime columns and ordering."""
    import pandas as pd

    relations = []
    for name, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            continue
        dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
        if dt_cols:
            relations.append(
                {"type": "temporal", "table": name, "datetime_columns": dt_cols}
            )
    return relations


# ---------------------------------------------------------------------------
# Main constructor (Algorithm 1, Appendix A.1)
# ---------------------------------------------------------------------------

class AdaptiveHypergraphConstructor:
    """
    Adaptive Hypergraph Constructor as per Algorithm 1 and Appendix A.1.

    Implements the four-level relation detection and hyperedge construction
    pipeline described in the paper.
    """

    def __init__(self, pruning_threshold: float = 0.1):
        self.pruning_threshold = pruning_threshold
        self.relations: dict[str, list[dict]] = {}

    def detect_relations_multi_level(
        self, tables: dict[str, Any]
    ) -> dict[str, list[dict]]:
        """
        Detect relations at four levels (Appendix A.1, lines 8-28):
          Level 1 — Explicit (foreign keys)
          Level 2 — Implicit (value overlaps)
          Level 3 — Semantic (metadata / NLP)
          Level 4 — Temporal (datetime patterns)
        """
        self.relations = {
            "explicit": _detect_foreign_keys(tables),
            "implicit": _detect_value_overlaps(tables),
            "semantic": _infer_semantic_relations(tables),
            "temporal": _detect_temporal_patterns(tables),
        }
        return self.relations

    def construct_hypergraph(
        self,
        tables: dict[str, Any],
        explicit_relations: list[tuple[str, str, str, str]] | None = None,
    ) -> RelationalHypergraph:
        """
        Build a RelationalHypergraph from the given tables.

        Parameters
        ----------
        tables : dict[str, DataFrame]
            Tables keyed by name.
        explicit_relations : list of (table1, key1, table2, key2), optional
            User-provided foreign-key definitions. If None, auto-detect.

        Returns
        -------
        RelationalHypergraph
        """
        import pandas as pd

        if not self.relations:
            self.detect_relations_multi_level(tables)

        # --- Extract entities (nodes) ---
        nodes: list[str] = []
        node_features: dict[str, np.ndarray] = {}
        for tname, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                continue
            for idx in df.index:
                node_id = f"{tname}:{idx}"
                nodes.append(node_id)
                numeric_cols = df.select_dtypes(include="number").columns
                feats = df.loc[idx, numeric_cols].values.astype(np.float32)
                node_features[node_id] = feats

        # --- Create n-ary hyperedges ---
        hyperedges: list[set[str]] = []
        weights: list[float] = []

        # From explicit FK relations: group records that share a key value
        fk_rels = explicit_relations or [
            (r["table1"], r["key"], r["table2"], r["key"])
            for r in self.relations.get("explicit", [])
        ]
        for t1, k1, t2, k2 in fk_rels:
            if t1 not in tables or t2 not in tables:
                continue
            df1, df2 = tables[t1], tables[t2]
            if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
                continue
            if k1 not in df1.columns or k2 not in df2.columns:
                continue
            shared_vals = set(df1[k1].dropna()) & set(df2[k2].dropna())
            for val in shared_vals:
                edge_nodes: set[str] = set()
                for idx in df1.index[df1[k1] == val]:
                    edge_nodes.add(f"{t1}:{idx}")
                for idx in df2.index[df2[k2] == val]:
                    edge_nodes.add(f"{t2}:{idx}")
                if len(edge_nodes) >= 2:
                    hyperedges.append(edge_nodes)
                    weights.append(1.0)

        hg = RelationalHypergraph(
            nodes=nodes,
            node_features=node_features,
            hyperedges=hyperedges,
            weights=weights,
            metadata={"relations": self.relations},
        )

        # --- Differentiable pruning (Appendix A.1, lines 43-55) ---
        hg = self.differentiable_pruning(hg)
        return hg

    def differentiable_pruning(self, hg: RelationalHypergraph) -> RelationalHypergraph:
        """
        Soft pruning of low-importance hyperedges (Appendix A.1, lines 43-55).

        In the full implementation this would use gradient-based importance scores;
        here we use edge-size heuristic as a proxy.
        """
        kept_edges = []
        kept_weights = []
        for edge, w in zip(hg.hyperedges, hg.weights):
            importance = w * len(edge)  # proxy: weight x arity
            if importance >= self.pruning_threshold:
                kept_edges.append(edge)
                kept_weights.append(w)
        hg.hyperedges = kept_edges
        hg.weights = kept_weights
        return hg
