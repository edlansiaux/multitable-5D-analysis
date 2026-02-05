import pytest
import pandas as pd
import numpy as np
import dgl
import torch
from mt5d.core.hypergraph import RelationalHypergraphBuilder, HyperEdge

class TestRelationalHypergraphBuilder:
    
    @pytest.fixture
    def sample_data(self):
        """Données de test pour construction d'hypergraphe"""
        patients = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [30, 40, 50],
            'diagnosis': ['I10', 'K37', 'I10']
        })
        
        visits = pd.DataFrame({
            'visit_id': [101, 102, 103],
            'patient_id': [1, 1, 2],
            'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-15'])
        })
        
        return {
            'patients': patients,
            'visits': visits
        }
    
    @pytest.fixture
    def sample_relationships(self):
        return [
            ('visits', 'patient_id', 'patients', 'patient_id', 'patient_visit')
        ]
    
    def test_builder_initialization(self):
        """Test l'initialisation du builder"""
        config = {'max_nodes_per_hyperedge': 50}
        builder = RelationalHypergraphBuilder(config)
        assert builder.config == config
        assert builder.hypergraph is None
    
    def test_create_nodes(self, sample_data):
        """Test la création de nœuds"""
        builder = RelationalHypergraphBuilder({})
        nodes = builder._create_nodes(sample_data)
        
        # Doit créer 3 patients + 3 visites = 6 nœuds
        assert len(nodes) == 6
        
        # Vérifie la structure des nœuds
        for node_id, node_info in nodes.items():
            assert 'features' in node_info
            assert 'type' in node_info
            assert 'table' in node_info
            assert 'original_index' in node_info
    
    def test_create_hyperedges_from_relationships(self, sample_data, sample_relationships):
        """Test la création d'hyperarêtes depuis les relations"""
        builder = RelationalHypergraphBuilder({})
        nodes = builder._create_nodes(sample_data)
        hyperedges = builder._create_hyperedges_from_relationships(
            sample_data, nodes, sample_relationships
        )
        
        # Doit créer des hyperarêtes pour les relations patient-visite
        assert len(hyperedges) > 0
        
        for hyperedge in hyperedges:
            assert isinstance(hyperedge, HyperEdge)
            assert hyperedge.id.startswith('patient_visit')
            assert len(hyperedge.nodes) >= 2
            assert hyperedge.type == 'patient_visit'
    
    def test_discover_implicit_relations(self, sample_data):
        """Test la découverte de relations implicites"""
        builder = RelationalHypergraphBuilder({})
        nodes = builder._create_nodes(sample_data)
        hyperedges = builder._discover_implicit_relations(sample_data, nodes)
        
        # Doit découvrir des relations par diagnostic similaire
        # Patients 1 et 3 ont le même diagnostic 'I10'
        found_diagnosis_relation = False
        for hyperedge in hyperedges:
            if 'diagnosis' in hyperedge.id:
                found_diagnosis_relation = True
                # Doit connecter les patients 1 et 3
                patient_nodes = [n for n in hyperedge.nodes if 'patients' in n]
                assert len(patient_nodes) >= 2
        
        assert found_diagnosis_relation
    
    def test_build_dgl_hypergraph(self, sample_data, sample_relationships):
        """Test la construction d'un hypergraphe DGL"""
        builder = RelationalHypergraphBuilder({})
        hypergraph = builder.build_from_tables(
            sample_data, sample_relationships
        )
        
        assert hypergraph is not None
        assert isinstance(hypergraph, dgl.DGLGraph)
        
        # Vérifie les nombres de nœuds et arêtes
        assert hypergraph.num_nodes('node') == 6
        assert hypergraph.num_edges('in') > 0
        
        # Vérifie la présence des features
        assert 'feat' in hypergraph.nodes['node'].data
        assert 'weight' in hypergraph.edges['in'].data
    
    def test_temporal_integration(self, sample_data):
        """Test l'intégration d'information temporelle"""
        config = {'temporal_integration': True}
        builder = RelationalHypergraphBuilder(config)
        
        temporal_columns = {
            'visits': ['date']
        }
        
        hypergraph = builder.build_from_tables(
            sample_data, 
            relationships=[],
            temporal_columns=temporal_columns
        )
        
        assert hypergraph is not None
        # À améliorer: vérifier que l'info temporelle est bien intégrée
    
    def test_empty_tables(self):
        """Test avec tables vides"""
        builder = RelationalHypergraphBuilder({})
        empty_tables = {'empty': pd.DataFrame()}
        
        with pytest.raises(Exception):
            builder.build_from_tables(empty_tables, [])
    
    def test_hyperedge_weights(self, sample_data, sample_relationships):
        """Test le calcul des poids des hyperarêtes"""
        config = {
            'explicit_relation_weight': 1.0,
            'implicit_relation_weight': 0.5
        }
        
        builder = RelationalHypergraphBuilder(config)
        hypergraph = builder.build_from_tables(sample_data, sample_relationships)
        
        # Les poids doivent être entre 0 et 1
        weights = hypergraph.edges['in'].data['weight']
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
