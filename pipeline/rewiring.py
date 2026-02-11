"""
Step 5: Dynamic Graph Rewiring (Section 5.5).

The model learns to reweight the relational graph based on downstream task
relevance, using attention weights as importance signals.
"""

from __future__ import annotations

import torch


class DynamicGraphRewirer:
    """Reweight adjacency using learned attention weights."""

    def __init__(self, top_k_ratio: float = 0.5, gate_threshold: float = 0.01):
        self.top_k_ratio = top_k_ratio
        self.gate_threshold = gate_threshold

    def rewire(
        self,
        adjacency: torch.Tensor,
        attention_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Produce a rewired adjacency matrix.

        Parameters
        ----------
        adjacency : (N, N)
        attention_weights : (H, N, N) or (B, H, N, N)
            Attention maps from the last layer.

        Returns
        -------
        Updated adjacency (N, N).
        """
        if attention_weights is None:
            return adjacency

        # Average across heads (and batch if present)
        w = attention_weights.float()
        while w.dim() > 2:
            w = w.mean(dim=0)

        # Gating: keep edges whose attention > threshold
        gate = (w > self.gate_threshold).float()
        rewired = adjacency * gate

        # Optionally prune to top-k per row
        k = max(1, int(rewired.shape[-1] * self.top_k_ratio))
        topk_vals, _ = rewired.topk(k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        rewired = rewired * (rewired >= threshold).float()

        return rewired
