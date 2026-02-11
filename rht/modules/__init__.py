"""Computational modules of the RHT (Section 4.2)."""

from rht.modules.hypergraph import AdaptiveHypergraphConstructor, RelationalHypergraph
from rht.modules.temporal import MultiScaleTemporalEmbedding
from rht.modules.attention import SparseRelationalAttention, HighCardinalityAttentionBlock
from rht.modules.discovery import DifferentiableRelationalDiscovery

__all__ = [
    "AdaptiveHypergraphConstructor",
    "RelationalHypergraph",
    "MultiScaleTemporalEmbedding",
    "SparseRelationalAttention",
    "HighCardinalityAttentionBlock",
    "DifferentiableRelationalDiscovery",
]
