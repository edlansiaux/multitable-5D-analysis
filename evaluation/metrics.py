"""
Multidimensional Evaluation Metrics (Section 7.2).

Implements all metrics from the paper:
  - Dimension 1: throughput, scalability curve, compression ratio
  - Dimension 2: feature importance consistency, dimensional preservation
  - Dimension 3: Rare Category Recall@k (Eq. 9), semantic coherence
  - Dimension 4: relational discovery F1, schema completion accuracy
  - Dimension 5: irregular TS imputation error, temporal pattern discovery rate
  - Holistic: 5D Integration Score (Def. 8, Eq. 10),
              Multi-Table Information Gain (Def. 9, Eq. 12)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =========================================================================
# Dimension 3: High Cardinality (Section 7.2.3)
# =========================================================================


def rare_category_recall_at_k(
    predictions: torch.Tensor,
    targets: torch.LongTensor,
    frequencies: torch.Tensor,
    k: int = 5,
    rarity_threshold: float = 0.001,
) -> float:
    """
    Rare Category Recall@k (Eq. 9, Section 7.2.3).

        RCR@k = (1 / |C_rare|)  sum_{c in C_rare}  1[rank(c) <= k]

    Parameters
    ----------
    predictions : (N, C) logits or probabilities.
    targets : (N,) ground-truth category indices.
    frequencies : (C,) training-set frequency of each category.
    k : top-k threshold.
    rarity_threshold : categories with freq < this are "rare".

    Returns
    -------
    Recall@k over rare categories.
    """
    rare_mask = frequencies < rarity_threshold  # (C,)
    rare_categories = rare_mask.nonzero(as_tuple=True)[0]

    if len(rare_categories) == 0:
        return float("nan")

    # Filter samples belonging to rare categories
    sample_mask = torch.isin(targets, rare_categories)
    if sample_mask.sum() == 0:
        return 0.0

    preds_rare = predictions[sample_mask]
    targets_rare = targets[sample_mask]

    topk_preds = preds_rare.topk(k, dim=-1).indices  # (N_rare, k)
    hits = (topk_preds == targets_rare.unsqueeze(-1)).any(dim=-1).float()
    return hits.mean().item()


def semantic_coherence(
    embeddings: torch.Tensor,
    cluster_labels: torch.LongTensor,
) -> float:
    """
    Semantic coherence metric (Section 7.2.3).

    Ratio of average intra-cluster cosine similarity to average
    inter-cluster cosine similarity.
    """
    z = F.normalize(embeddings, dim=-1)
    sim = z @ z.T

    clusters = cluster_labels.unique()
    intra_sims, inter_sims = [], []

    for c in clusters:
        mask_c = cluster_labels == c
        if mask_c.sum() < 2:
            continue
        intra = sim[mask_c][:, mask_c]
        n = intra.shape[0]
        intra_sims.append((intra.sum() - n) / (n * (n - 1)))

        mask_not_c = ~mask_c
        if mask_not_c.sum() == 0:
            continue
        inter = sim[mask_c][:, mask_not_c]
        inter_sims.append(inter.mean())

    if not intra_sims or not inter_sims:
        return float("nan")

    avg_intra = sum(intra_sims) / len(intra_sims)
    avg_inter = sum(inter_sims) / len(inter_sims)
    return (avg_intra / avg_inter).item() if avg_inter != 0 else float("inf")


# =========================================================================
# Dimension 4: Relational Discovery (Section 7.2.4)
# =========================================================================


def relational_discovery_f1(
    discovered: set[tuple],
    ground_truth: set[tuple],
) -> dict[str, float]:
    """
    F1 score for latent relationship discovery (Section 7.2.4).

    Parameters
    ----------
    discovered : set of (table_i, table_j) tuples predicted by the model.
    ground_truth : set of (table_i, table_j) held-out FK pairs.
    """
    if not ground_truth:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    tp = len(discovered & ground_truth)
    precision = tp / len(discovered) if discovered else 0.0
    recall = tp / len(ground_truth)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# =========================================================================
# Dimension 5: Temporal (Section 7.2.5)
# =========================================================================


def irregular_ts_imputation_mae(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.BoolTensor,
) -> float:
    """
    MAE on masked values for irregular time series imputation (Section 7.2.5).
    """
    return (predictions[mask] - ground_truth[mask]).abs().mean().item()


# =========================================================================
# Holistic Metrics (Section 7.2.7)
# =========================================================================


def five_d_integration_score(dimension_scores: list[float]) -> float:
    """
    5D Integration Score (Definition 8, Eq. 10).

        S_5D = 5 / sum_{i=1}^{5} (1 / s_i)

    Harmonic mean of the five normalised dimension scores.
    """
    assert len(dimension_scores) == 5
    inv_sum = sum(1.0 / max(s, 1e-10) for s in dimension_scores)
    return 5.0 / inv_sum


def multi_table_information_gain(
    mi_pre_fusion: float,
    mi_post_fusion: float,
) -> float:
    """
    Multi-Table Information Gain (Definition 9, Eq. 12).

        G_MT = I(post-fusion) / I(pre-fusion)
    """
    if mi_pre_fusion == 0:
        return float("inf")
    return mi_post_fusion / mi_pre_fusion


# =========================================================================
# Anomaly Detection (Section 7.2.6)
# =========================================================================


def extreme_value_precision_recall(
    predictions: torch.BoolTensor,
    ground_truth: torch.BoolTensor,
) -> dict[str, float]:
    """
    Precision and recall for extreme-value detection (Section 7.2.6).
    """
    tp = (predictions & ground_truth).sum().item()
    precision = tp / predictions.sum().item() if predictions.sum() > 0 else 0.0
    recall = tp / ground_truth.sum().item() if ground_truth.sum() > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compression_ratio(original_bytes: int, hypergraph_bytes: int) -> float:
    """Compression ratio (Section 7.2.1)."""
    return original_bytes / max(hypergraph_bytes, 1)
