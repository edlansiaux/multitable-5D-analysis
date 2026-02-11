"""
Step 7: Federated Multi-Table Learning (Section 5.7).

Strategy:
  - Local learning on schema fragments
  - Secure aggregation of relational embeddings
  - Differential privacy at relationship level
"""

from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn as nn


def federated_average(
    global_model: nn.Module,
    local_models: list[nn.Module],
    weights: list[float] | None = None,
) -> nn.Module:
    """
    FedAvg aggregation of local model parameters.

    Parameters
    ----------
    global_model : the global model whose parameters will be updated.
    local_models : list of locally trained models.
    weights : optional per-client weighting (e.g. proportional to data size).
    """
    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)

    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(
            w * lm.state_dict()[key].float()
            for w, lm in zip(weights, local_models)
        )
    global_model.load_state_dict(global_dict)
    return global_model


def add_differential_privacy_noise(
    parameters: dict[str, torch.Tensor],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Add calibrated Gaussian noise for (epsilon, delta)-differential privacy
    at relationship level (Section 5.7, bullet 3).
    """
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noisy = {}
    for key, param in parameters.items():
        noisy[key] = param + torch.randn_like(param) * sigma
    return noisy
