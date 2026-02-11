"""
Module 4: Differentiable Relational Discovery.

Implements Definition 4 (Relational Discovery Loss) from Section 4.2.4
and the causal relational discovery methods from Section 6.4.

    L_rel = alpha * L_task  +  beta * L_sparse  +  gamma * L_semantic
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableRelationalDiscovery(nn.Module):
    """
    Learns to discover latent relationships not explicit in the schema.

    The module produces an auxiliary loss that encourages:
      - L_sparse: the discovered graph to remain sparse
      - L_semantic: semantically coherent relationships

    L_task is provided externally (the supervised objective).
    """

    def __init__(
        self,
        embed_dim: int,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Bilinear relation scorer: score(i,j) = zi^T . R . zj
        self.relation_matrix = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.relation_matrix)

        # Semantic coherence projector
        self.semantic_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
        )

    def _compute_relation_scores(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise relation scores.

        Parameters
        ----------
        z : (N, D) node embeddings.

        Returns
        -------
        scores : (N, N) relation probability logits.
        """
        # z @ R @ z^T
        Rz = z @ self.relation_matrix     # (N, D)
        scores = Rz @ z.T                  # (N, N)
        return scores

    def sparsity_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """L_sparse: penalise excessive graph density (L1 on sigmoid scores)."""
        probs = torch.sigmoid(scores)
        return probs.mean()

    def semantic_loss(self, z: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        L_semantic: encourage discovered edges to connect semantically similar nodes.

        We project embeddings to a low-dim semantic space and penalise
        high-scoring edges between dissimilar nodes.
        """
        sem = self.semantic_proj(z)                           # (N, d')
        # Cosine similarity in semantic space
        sem_norm = F.normalize(sem, dim=-1)
        sem_sim = sem_norm @ sem_norm.T                       # (N, N)

        probs = torch.sigmoid(scores)
        # We want high-prob edges to correspond to high semantic similarity
        loss = -((probs * sem_sim).mean())
        return loss

    def forward(
        self,
        z: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the combined auxiliary discovery loss (Eq. 4, without L_task).

        The caller should add alpha * L_task to the returned value to obtain
        the full L_rel.
        """
        if z.dim() == 3:
            z = z.squeeze(0)

        scores = self._compute_relation_scores(z)
        l_sparse = self.sparsity_loss(scores)
        l_semantic = self.semantic_loss(z, scores)
        return self.beta * l_sparse + self.gamma * l_semantic

    def discover_relations(
        self, z: torch.Tensor, threshold: float = 0.5
    ) -> list[tuple[int, int, float]]:
        """
        Return a list of discovered (latent) relations above threshold.

        Returns
        -------
        list of (node_i, node_j, score)
        """
        if z.dim() == 3:
            z = z.squeeze(0)
        scores = torch.sigmoid(self._compute_relation_scores(z))
        mask = scores > threshold
        indices = mask.nonzero(as_tuple=False)
        results = []
        for row in indices:
            i, j = row[0].item(), row[1].item()
            if i < j:
                results.append((i, j, scores[i, j].item()))
        return results
