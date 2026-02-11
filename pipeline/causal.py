"""
Step 6: Relational Causal Inference (Section 5.6 / 6.4).

Methods:
  - Double Machine Learning for causal effect estimation
  - Granger causality tests adapted to graphs
  - Structural Equation Modeling on hypergraphs
"""

from __future__ import annotations

import torch


def granger_causality_graph(
    time_series: dict[str, torch.Tensor],
    adjacency: torch.Tensor,
    max_lag: int = 5,
    significance: float = 0.01,
) -> list[dict]:
    """
    Granger causality tests adapted to the relational graph (Section 6.4).

    For each directed edge (u -> v) in the graph, test whether the past of u
    helps predict v beyond v's own past.

    Parameters
    ----------
    time_series : dict mapping node id -> (T,) tensor
    adjacency : (N, N) adjacency matrix
    max_lag : maximum lag to consider
    significance : p-value threshold

    Returns
    -------
    List of dicts with causal links found.
    """
    # Placeholder — full implementation requires VAR fitting + F-tests
    # See Section 6.4 item 1.
    results = []
    node_ids = list(time_series.keys())
    for i, u in enumerate(node_ids):
        for j, v in enumerate(node_ids):
            if i == j:
                continue
            if adjacency[i, j] > 0:
                results.append(
                    {
                        "cause": u,
                        "effect": v,
                        "lag": max_lag,
                        "p_value": None,  # to be computed
                        "status": "pending",
                    }
                )
    return results


def relational_counterfactual(
    embeddings: torch.Tensor,
    adjacency: torch.Tensor,
    edge_to_remove: tuple[int, int],
) -> torch.Tensor:
    """
    Relational counterfactual: "What if this relationship didn't exist?"
    (Section 5.6, bullet 2).

    Removes the specified edge and recomputes embeddings to estimate
    the causal effect of the relationship.
    """
    adj_cf = adjacency.clone()
    i, j = edge_to_remove
    adj_cf[i, j] = 0
    adj_cf[j, i] = 0
    # The caller should re-run the model with adj_cf
    return adj_cf
