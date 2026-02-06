import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class PentE(nn.Module):
    """
    Pentadimensional Embedding (Section 5.3 & Eq 5)
    Intègre : Sémantique, Relationnel, Temporel, Catégoriel, Volumétrique.
    """
    
    def __init__(self, 
                 node_dim: int = 128,
                 relation_dim: int = 64,
                 temporal_dim: int = 32,
                 categorical_dim: int = 64,
                 volume_dim: int = 16,
                 output_dim: int = 256,
                 use_attention: bool = True):
        super().__init__()
        
        # 1. Dimension Sémantique (Attributs)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )
        
        # 2. Dimension Relationnelle (Position structurelle)
        self.relation_encoder = nn.Linear(relation_dim, output_dim)
        
        # 3. Dimension Temporelle (Time2Vec)
        self.temporal_encoder = TemporalEncoder(temporal_dim, output_dim)
        
        # 4. Dimension Catégorielle (Haute Cardinalité)
        self.categorical_encoder = HighCardinalityEncoder(categorical_dim, output_dim)
        
        # 5. Dimension Volumétrique (Stats de volume)
        self.volume_encoder = VolumeEncoder(volume_dim, output_dim)
        
        # Mécanisme de fusion (Concaténation ou Attention)
        self.use_attention = use_attention
        if use_attention:
            self.dimension_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True
            )
        
        # Projection finale z_e
        self.combiner = nn.Sequential(
            nn.Linear(output_dim * 5 if not use_attention else output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, 
                node_features: torch.Tensor,
                relation_features: torch.Tensor,
                temporal_features: torch.Tensor,
                categorical_features: Dict[str, torch.Tensor],
                volume_features: torch.Tensor) -> torch.Tensor:
        
        # Encodage individuel des 5 dimensions
        h_sem = self.node_encoder(node_features)
        h_rel = self.relation_encoder(relation_features)
        h_temp = self.temporal_encoder(temporal_features)
        h_cat = self.categorical_encoder(categorical_features)
        h_vol = self.volume_encoder(volume_features)
        
        if self.use_attention:
            # Stack pour attention: [batch, 5, dim]
            dimensions = torch.stack([h_sem, h_rel, h_temp, h_cat, h_vol], dim=1)
            attended, _ = self.dimension_attention(dimensions, dimensions, dimensions)
            combined = attended.mean(dim=1)
        else:
            # Concaténation (Eq 5)
            combined = torch.cat([h_sem, h_rel, h_temp, h_cat, h_vol], dim=-1)
        
        return self.combiner(combined)

class VolumeEncoder(nn.Module):
    """Encodeur pour la dimension volumétrique (stats d'agrégation)"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HighCardinalityEncoder(nn.Module):
    """
    Section 6.2: Hierarchical High-Cardinality Encoding
    Gère l'embedding composite : code + cluster + parent
    """
    def __init__(self, embedding_dim: int, output_dim: int, num_embeddings: int = 10000):
        super().__init__()
        # Embedding de base pour les codes
        self.code_embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Embedding pour les clusters/parents (simplifié ici par un seul niveau hiérarchique)
        self.hierarchy_embedding = nn.Embedding(num_embeddings // 10, embedding_dim)
        
        self.projection = nn.Linear(embedding_dim * 2, output_dim)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # On attend 'indices' et 'parent_indices' dans le dict features
        indices = features.get("indices", torch.zeros(1, dtype=torch.long))
        parents = features.get("parents", torch.zeros(1, dtype=torch.long))
        
        # Eq 7: e_c = e_code + e_parent (simplifié ici en concat)
        e_code = self.code_embedding(indices)
        e_parent = self.hierarchy_embedding(parents)
        
        return self.projection(torch.cat([e_code, e_parent], dim=-1))

class TemporalEncoder(nn.Module):
    """Section 4.2.2: Multi-Scale Temporal Embeddings (Time2Vec)"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        # Composante linéaire (wot)
        self.w0 = nn.Linear(input_dim, output_dim // 2) 
        # Composante périodique (sin(wt))
        self.w_periodic = nn.Linear(input_dim, output_dim // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Definition 5
        linear = self.w0(x)
        periodic = torch.sin(self.w_periodic(x))
        return torch.cat([linear, periodic], dim=-1)
