import pytest
import torch
import torch.nn as nn
from mt5d.models.embeddings import PentE, HighCardinalityEncoder

class TestPentE:
    
    @pytest.fixture
    def sample_features(self):
        """Crée des features de test pour PentE"""
        batch_size = 16
        
        return {
            'node_features': torch.randn(batch_size, 128),
            'relation_features': torch.randn(batch_size, 64),
            'temporal_features': torch.randn(batch_size, 32),
            'categorical_features': {
                'categorical_indices': torch.randint(0, 1000, (batch_size,)),
                'continuous_features': torch.randn(batch_size, 10)
            },
            'volume_features': torch.randn(batch_size, 16)
        }
    
    def test_pente_initialization(self):
        """Test l'initialisation de PentE"""
        model = PentE(
            node_dim=128,
            relation_dim=64,
            temporal_dim=32,
            categorical_dim=64,
            volume_dim=16,
            output_dim=256,
            use_attention=True
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'node_encoder')
        assert hasattr(model, 'categorical_encoder')
        assert hasattr(model, 'dimension_attention')
    
    def test_pente_forward(self, sample_features):
        """Test le forward pass de PentE"""
        model = PentE(output_dim=256)
        
        output = model(
            sample_features['node_features'],
            sample_features['relation_features'],
            sample_features['temporal_features'],
            sample_features['categorical_features'],
            sample_features['volume_features']
        )
        
        assert output.shape == torch.Size([16, 256])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_pente_without_attention(self, sample_features):
        """Test PentE sans mécanisme d'attention"""
        model = PentE(output_dim=256, use_attention=False)
        
        output = model(
            sample_features['node_features'],
            sample_features['relation_features'],
            sample_features['temporal_features'],
            sample_features['categorical_features'],
            sample_features['volume_features']
        )
        
        assert output.shape == torch.Size([16, 256])
    
    def test_pente_different_batch_sizes(self):
        """Test PentE avec différentes tailles de batch"""
        model = PentE(output_dim=128)
        
        for batch_size in [1, 8, 32, 128]:
            features = {
                'node_features': torch.randn(batch_size, 128),
                'relation_features': torch.randn(batch_size, 64),
                'temporal_features': torch.randn(batch_size, 32),
                'categorical_features': {'batch_size': batch_size},
                'volume_features': torch.randn(batch_size, 16)
            }
            
            output = model(**features)
            assert output.shape == torch.Size([batch_size, 128])
    
    def test_pente_gradient_flow(self, sample_features):
        """Test que les gradients circulent correctement"""
        model = PentE(output_dim=128)
        
        # Forward pass
        output = model(
            sample_features['node_features'],
            sample_features['relation_features'],
            sample_features['temporal_features'],
            sample_features['categorical_features'],
            sample_features['volume_features']
        )
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Vérifie que tous les paramètres ont des gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

class TestHighCardinalityEncoder:
    
    def test_high_cardinality_encoder_initialization(self):
        """Test l'initialisation de l'encodeur haute cardinalité"""
        encoder = HighCardinalityEncoder(input_dim=10, output_dim=64)
        
        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, 'embedding')
        assert encoder.num_buckets == 10000
    
    def test_high_cardinality_forward_with_indices(self):
        """Test le forward avec indices catégoriels"""
        encoder = HighCardinalityEncoder(input_dim=10, output_dim=64)
        
        batch_size = 16
        features = {
            'categorical_indices': torch.randint(0, 1000, (batch_size,)),
            'continuous_features': torch.randn(batch_size, 10)
        }
        
        output = encoder(features)
        assert output.shape == torch.Size([batch_size, 64])
    
    def test_high_cardinality_forward_without_indices(self):
        """Test le forward sans indices catégoriels"""
        encoder = HighCardinalityEncoder(input_dim=10, output_dim=64)
        
        batch_size = 16
        features = {
            'continuous_features': torch.randn(batch_size, 10)
        }
        
        output = encoder(features)
        assert output.shape == torch.Size([batch_size, 64])
    
    def test_high_cardinality_large_buckets(self):
        """Test avec un grand nombre de buckets"""
        encoder = HighCardinalityEncoder(
            input_dim=10, 
            output_dim=64,
            num_buckets=50000
        )
        
        indices = torch.randint(0, 50000, (32,))
        features = {'categorical_indices': indices}
        
        output = encoder(features)
        assert output.shape == torch.Size([32, 64])
