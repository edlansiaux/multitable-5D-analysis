import pytest
import torch
import dgl
from mt5d.models.architectures import RelationalHypergraphTransformer

class TestRelationalHypergraphTransformer:
    
    @pytest.fixture
    def sample_hypergraph(self):
        """Crée un hypergraphe de test"""
        # Crée un graphe simple
        g = dgl.heterograph({
            ('node', 'in', 'hyperedge'): (
                torch.tensor([0, 1, 2, 3, 4]),
                torch.tensor([0, 0, 1, 1, 2])
            )
        })
        
        # Ajoute des features
        g.nodes['node'].data['feat'] = torch.randn(5, 128)
        g.edges['in'].data['weight'] = torch.tensor([1.0, 0.8, 0.6, 0.9, 0.7])
        g.edges['in'].data['type'] = torch.tensor([0, 0, 1, 1, 2])
        
        return g
    
    def test_rht_initialization(self):
        """Test l'initialisation du RHT"""
        model = RelationalHypergraphTransformer(
            input_dim=128,
            hidden_dim=256,
            output_dim=64,
            num_heads=8,
            num_layers=3,
            use_pente=True
        )
        
        assert isinstance(model, torch.nn.Module)
        assert model.use_pente == True
        assert len(model.layers) == 3
    
    def test_rht_forward_with_pente(self, sample_hypergraph):
        """Test le forward pass avec PentE"""
        model = RelationalHypergraphTransformer(
            input_dim=128,
            hidden_dim=256,
            output_dim=64,
            use_pente=True
        )
        
        node_features = sample_hypergraph.nodes['node'].data['feat']
        edge_features = sample_hypergraph.edges['in'].data['weight']
        
        output = model(sample_hypergraph, node_features, edge_features)
        
        assert output.shape == torch.Size([5, 64])
        assert not torch.isnan(output).any()
    
    def test_rht_forward_without_pente(self, sample_hypergraph):
        """Test le forward pass sans PentE"""
        model = RelationalHypergraphTransformer(
            input_dim=128,
            hidden_dim=256,
            output_dim=64,
            use_pente=False
        )
        
        node_features = sample_hypergraph.nodes['node'].data['feat']
        output = model(sample_hypergraph, node_features)
        
        assert output.shape == torch.Size([5, 64])
    
    def test_rht_different_graph_sizes(self):
        """Test avec différentes tailles de graphe"""
        model = RelationalHypergraphTransformer(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        )
        
        for num_nodes in [1, 10, 100]:
            # Crée un graphe de taille variable
            g = dgl.heterograph({
                ('node', 'in', 'hyperedge'): (
                    torch.arange(num_nodes),
                    torch.arange(num_nodes) % 3
                )
            })
            g.nodes['node'].data['feat'] = torch.randn(num_nodes, 128)
            
            output = model(g, g.nodes['node'].data['feat'])
            assert output.shape == torch.Size([num_nodes, 64])
    
    def test_rht_with_temporal_info(self, sample_hypergraph):
        """Test avec information temporelle"""
        model = RelationalHypergraphTransformer(
            input_dim=128,
            hidden_dim=256,
            output_dim=64,
            use_pente=True
        )
        
        node_features = sample_hypergraph.nodes['node'].data['feat']
        temporal_info = torch.randn(5, 32)  # 5 nœuds, 32 features temporelles
        
        output = model(sample_hypergraph, node_features, temporal_info=temporal_info)
        assert output.shape == torch.Size([5, 64])
    
    def test_rht_gradient_flow(self, sample_hypergraph):
        """Test la circulation des gradients"""
        model = RelationalHypergraphTransformer(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        )
        
        node_features = sample_hypergraph.nodes['node'].data['feat']
        node_features.requires_grad = True
        
        output = model(sample_hypergraph, node_features)
        
        # Calcul de la loss et backward
        loss = output.sum()
        loss.backward()
        
        # Vérifie les gradients
        assert node_features.grad is not None
        assert not torch.isnan(node_features.grad).any()
        
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any()
