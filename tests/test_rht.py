import torch
import dgl
import pytest
from mt5d.models.architectures.rht import RelationalHypergraphTransformer

def test_rht_forward_shape():
    """
    Vérifie que le RHT accepte un graphe et des features, 
    et retourne un tenseur de la bonne dimension.
    """
    # 1. Création d'un mini-graphe factice
    num_nodes = 10
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 0])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    # 2. Features factices (5 dimensions)
    input_dim = 16
    hidden_dim = 32
    output_dim = 5
    
    node_feats = torch.randn(num_nodes, input_dim)
    edge_feats = torch.randn(4, 64) # Relation dim par défaut
    temp_feats = torch.randn(num_nodes, 32) # Temp dim par défaut
    
    # 3. Initialisation Modèle
    model = RelationalHypergraphTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=2,
        use_pente=True
    )
    
    # 4. Forward Pass
    # Note: RHT attend des kwargs pour les dimensions optionnelles
    out = model(
        g, 
        node_features=node_feats,
        edge_features=edge_feats,
        temporal_features=temp_feats
    )
    
    # 5. Assertions
    assert out.shape == (num_nodes, output_dim)
    assert not torch.isnan(out).any()

def test_rht_without_pente():
    """Vérifie le fonctionnement sans l'embedding 5D (Ablation study)."""
    model = RelationalHypergraphTransformer(
        input_dim=16,
        hidden_dim=32,
        output_dim=2,
        use_pente=False
    )
    g = dgl.graph(([0], [1]), num_nodes=2)
    node_feats = torch.randn(2, 16)
    
    out = model(g, node_features=node_feats)
    assert out.shape == (2, 2)
