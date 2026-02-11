"""
Relational Discovery Loss wrapper (Definition 4, Section 4.2.4).

    L_rel = alpha * L_task + beta * L_sparse + gamma * L_semantic

This module wraps any task loss with the discovery auxiliary terms
computed by Module 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationalDiscoveryLoss(nn.Module):
    """Combined supervised + discovery loss."""

    def __init__(self, task_loss_fn: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        discovery_loss: torch.Tensor,
    ) -> torch.Tensor:
        l_task = self.task_loss_fn(logits, targets)
        return self.alpha * l_task + discovery_loss
