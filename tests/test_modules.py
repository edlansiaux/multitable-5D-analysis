"""
Unit tests for the core RHT modules.

Run with:  pytest tests/test_modules.py -v
"""

from __future__ import annotations

import pytest
import torch
import pandas as pd
import numpy as np

from rht.model import RelationalHypergraphTransformer, RHTConfig
from rht.modules.hypergraph import AdaptiveHypergraphConstructor, RelationalHypergraph
from rht.modules.temporal import Time2Vec, MultiScaleTemporalEmbedding
from rht.modules.attention import SparseRelationalAttention, RareCategoryMemoryBank
from rht.modules.discovery import DifferentiableRelationalDiscovery
from rht.embeddings.pente import PentEEmbedding
from rht.embeddings.categorical import HierarchicalCategoricalEncoder
from rht.losses.contrastive import RelationalTemporalContrastiveLoss
from evaluation.metrics import (
    rare_category_recall_at_k,
    five_d_integration_score,
    relational_discovery_f1,
    semantic_coherence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tables():
    """Create small sample tables mimicking a medical schema."""
    patients = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [45, 67, 32],
            "weight": [70.5, 82.0, 55.3],
        }
    )
    admissions = pd.DataFrame(
        {
            "hadm_id": [100, 101, 102],
            "patient_id": [1, 2, 3],
            "los_days": [3, 7, 2],
        }
    )
    diagnoses = pd.DataFrame(
        {
            "hadm_id": [100, 100, 101, 102],
            "icd_code": [250, 401, 250, 410],
            "seq_num": [1, 2, 1, 1],
        }
    )
    return {
        "patients": patients,
        "admissions": admissions,
        "diagnoses": diagnoses,
    }


@pytest.fixture
def sample_relations():
    return [
        ("patients", "patient_id", "admissions", "patient_id"),
        ("admissions", "hadm_id", "diagnoses", "hadm_id"),
    ]


@pytest.fixture
def small_config():
    return RHTConfig(
        semantic_dim=16,
        relational_dim=8,
        temporal_dim=8,
        categorical_dim=8,
        volume_dim=4,
        num_attention_heads=2,
        num_message_passing_layers=1,
    )


# ---------------------------------------------------------------------------
# Module 1: Hypergraph Construction
# ---------------------------------------------------------------------------


class TestHypergraphConstruction:
    def test_construct_from_tables(self, sample_tables, sample_relations):
        constructor = AdaptiveHypergraphConstructor()
        hg = constructor.construct_hypergraph(sample_tables, sample_relations)
        assert isinstance(hg, RelationalHypergraph)
        assert hg.num_nodes > 0
        assert hg.num_edges > 0

    def test_incidence_matrix_shape(self, sample_tables, sample_relations):
        constructor = AdaptiveHypergraphConstructor()
        hg = constructor.construct_hypergraph(sample_tables, sample_relations)
        H = hg.incidence_matrix()
        assert H.shape == (hg.num_nodes, hg.num_edges)

    def test_adjacency_symmetric(self, sample_tables, sample_relations):
        constructor = AdaptiveHypergraphConstructor()
        hg = constructor.construct_hypergraph(sample_tables, sample_relations)
        A = hg.adjacency_from_incidence()
        assert torch.allclose(A, A.T)

    def test_multi_level_detection(self, sample_tables):
        constructor = AdaptiveHypergraphConstructor()
        rels = constructor.detect_relations_multi_level(sample_tables)
        assert "explicit" in rels
        assert "implicit" in rels
        assert "semantic" in rels
        assert "temporal" in rels


# ---------------------------------------------------------------------------
# Module 2: Temporal Embeddings
# ---------------------------------------------------------------------------


class TestTemporalEmbeddings:
    def test_time2vec_shape(self):
        t2v = Time2Vec(out_dim=16)
        t = torch.tensor([0.0, 1.0, 100.0])
        out = t2v(t)
        assert out.shape == (3, 16)

    def test_multiscale_shape(self):
        enc = MultiScaleTemporalEmbedding(per_scale_dim=8, out_dim=32)
        t = torch.randn(10)
        out = enc(t)
        assert out.shape == (10, 32)


# ---------------------------------------------------------------------------
# Module 3: Sparse Relational Attention
# ---------------------------------------------------------------------------


class TestSparseAttention:
    def test_output_shape(self):
        attn = SparseRelationalAttention(embed_dim=32, num_heads=4)
        x = torch.randn(5, 32)
        out, w = attn(x)
        assert out.shape == (5, 32)

    def test_masking(self):
        attn = SparseRelationalAttention(embed_dim=16, num_heads=2)
        x = torch.randn(4, 16)
        adj = torch.eye(4)
        adj[0, 1] = adj[1, 0] = 1
        out, w = attn(x, adjacency=adj)
        assert out.shape == x.shape


class TestMemoryBank:
    def test_update_and_classify(self):
        bank = RareCategoryMemoryBank(embed_dim=16, max_categories=10)
        emb = torch.randn(5, 16)
        labels = torch.tensor([0, 1, 2, 0, 1])
        bank.update(emb, labels)
        preds = bank.classify(emb)
        assert preds.shape == (5,)


# ---------------------------------------------------------------------------
# Module 4: Relational Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_loss_scalar(self):
        disc = DifferentiableRelationalDiscovery(embed_dim=16)
        z = torch.randn(10, 16)
        loss = disc(z)
        assert loss.dim() == 0

    def test_discover_relations(self):
        disc = DifferentiableRelationalDiscovery(embed_dim=16)
        z = torch.randn(10, 16)
        rels = disc.discover_relations(z, threshold=0.3)
        assert isinstance(rels, list)


# ---------------------------------------------------------------------------
# PentE Embedding
# ---------------------------------------------------------------------------


class TestPentE:
    def test_output_dim(self, small_config):
        pente = PentEEmbedding(small_config)
        x = torch.randn(5, 10)
        out = pente(node_features=x)
        assert out.shape == (5, small_config.pente_dim)


# ---------------------------------------------------------------------------
# Categorical Encoder
# ---------------------------------------------------------------------------


class TestCategorical:
    def test_composite_shape(self):
        enc = HierarchicalCategoricalEncoder(embed_dim=12, num_codes=100)
        codes = torch.randint(0, 100, (8,))
        out = enc(codes)
        assert out.shape == (8, 12)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestContrastiveLoss:
    def test_forward(self):
        loss_fn = RelationalTemporalContrastiveLoss()
        z = torch.randn(10, 32)
        adj = torch.eye(10)
        adj[0, 1] = adj[1, 0] = 1
        loss = loss_fn(z, adj)
        assert loss.dim() == 0
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_rcr_at_k(self):
        preds = torch.randn(10, 50)
        targets = torch.randint(0, 50, (10,))
        freqs = torch.rand(50) * 0.01  # all rare
        result = rare_category_recall_at_k(preds, targets, freqs, k=5)
        assert 0.0 <= result <= 1.0

    def test_5d_score(self):
        scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        s = five_d_integration_score(scores)
        assert 0 < s <= 1.0

    def test_relational_f1(self):
        discovered = {(0, 1), (1, 2), (3, 4)}
        gt = {(0, 1), (1, 2), (2, 3)}
        result = relational_discovery_f1(discovered, gt)
        assert 0 <= result["f1"] <= 1.0

    def test_semantic_coherence(self):
        z = torch.randn(20, 16)
        labels = torch.randint(0, 3, (20,))
        sc = semantic_coherence(z, labels)
        assert isinstance(sc, float)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class TestRHTModel:
    def test_forward(self, small_config):
        model = RelationalHypergraphTransformer(small_config)
        x = {"table_a": torch.randn(5, 10)}
        out = model(node_features=x)
        assert "embeddings" in out
        assert "logits" in out
        assert "discovery_loss" in out
        assert out["embeddings"].shape[0] == 5
        assert out["embeddings"].shape[1] == small_config.pente_dim
