"""
Relational-Temporal Contrastive Loss (Definition 6, Section 5.4).

    L_CRT = alpha * L_rel + beta * L_temp + gamma * L_sem

where:
  - L_rel  brings together linked entities, separates unlinked ones
  - L_temp aligns temporally close entities
  - L_sem  encourages semantic coherence
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalTemporalContrastiveLoss(nn.Module):
    """
    Combined contrastive loss for Step 4 (relational contrastive learning).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def _info_nce(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE loss."""
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)

        pos_sim = (anchors * positives).sum(dim=-1) / self.temperature

        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = (
                (anchors.unsqueeze(1) * negatives.unsqueeze(0)).sum(dim=-1)
                / self.temperature
            )
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        else:
            # All other examples in the batch are negatives
            all_sim = anchors @ positives.T / self.temperature
            logits = all_sim

        labels = torch.zeros(
            anchors.shape[0], dtype=torch.long, device=anchors.device
        )
        return F.cross_entropy(logits, labels)

    def relational_loss(
        self,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """L_rel: linked entities should be close, unlinked far."""
        z = F.normalize(embeddings, dim=-1)
        sim = z @ z.T / self.temperature
        # Positive pairs: adjacency > 0
        pos_mask = (adjacency > 0).float()
        neg_mask = 1.0 - pos_mask
        pos_mask.fill_diagonal_(0)
        neg_mask.fill_diagonal_(0)

        # Contrastive: maximise similarity of positives vs negatives
        pos_sim = (sim * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)
        neg_sim = (sim * neg_mask).sum(dim=-1) / neg_mask.sum(dim=-1).clamp(min=1)
        return F.relu(neg_sim - pos_sim + 0.5).mean()

    def temporal_loss(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        threshold: float = 3600.0,
    ) -> torch.Tensor:
        """L_temp: temporally close entities should have similar embeddings."""
        z = F.normalize(embeddings, dim=-1)
        time_diff = (timestamps.unsqueeze(0) - timestamps.unsqueeze(1)).abs()
        close_mask = (time_diff < threshold).float()
        close_mask.fill_diagonal_(0)
        far_mask = (time_diff >= threshold).float()

        sim = z @ z.T
        close_sim = (sim * close_mask).sum(dim=-1) / close_mask.sum(dim=-1).clamp(
            min=1
        )
        far_sim = (sim * far_mask).sum(dim=-1) / far_mask.sum(dim=-1).clamp(min=1)
        return F.relu(far_sim - close_sim + 0.5).mean()

    def semantic_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L_sem: regularisation for embedding smoothness."""
        z = F.normalize(embeddings, dim=-1)
        cov = z.T @ z / z.shape[0]
        # Off-diagonal should be small -> decorrelation
        off_diag = cov - torch.diag(cov.diag())
        return off_diag.pow(2).mean()

    def forward(
        self,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute L_CRT (Eq. 6)."""
        l_rel = self.relational_loss(embeddings, adjacency)
        l_temp = torch.tensor(0.0, device=embeddings.device)
        if timestamps is not None:
            l_temp = self.temporal_loss(embeddings, timestamps)
        l_sem = self.semantic_loss(embeddings)
        return self.alpha * l_rel + self.beta * l_temp + self.gamma * l_sem
