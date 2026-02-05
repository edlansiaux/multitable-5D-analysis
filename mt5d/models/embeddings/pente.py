import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class PentE(nn.Module):
    """
    Pentadimensional Embedding: Combinaison de 5 dimensions
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
        
        # Encodeurs pour chaque dimension
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )
        
        self.relation_encoder = RelationEncoder(relation_dim, output_dim)
        self.temporal_encoder = TemporalEncoder(temporal_dim, output_dim)
        self.categorical_encoder = HighCardinalityEncoder(categorical_dim, output_dim)
        self.volume_encoder = VolumeEncoder(volume_dim, output_dim)
        
        # Mécanisme d'attention pour combinaison
        self.use_attention = use_attention
        if use_attention:
            self.dimension_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True
            )
        
        # Combinaison finale
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
                categorical_features: torch.Tensor,
                volume_features: torch.Tensor) -> torch.Tensor:
        """Forward pass pour PentE"""
        
        # Encoder chaque dimension
        h_node = self.node_encoder(node_features)
        h_rel = self.relation_encoder(relation_features)
        h_time = self.temporal_encoder(temporal_features)
        h_cat = self.categorical_encoder(categorical_features)
        h_vol = self.volume_encoder(volume_features)
        
        if self.use_attention:
            # Combinaison par attention
            dimensions = torch.stack([h_node, h_rel, h_time, h_cat, h_vol], dim=1)
            attended, _ = self.dimension_attention(dimensions, dimensions, dimensions)
            combined = attended.mean(dim=1)
        else:
            # Concatenation simple
            combined = torch.cat([h_node, h_rel, h_time, h_cat, h_vol], dim=-1)
        
        # Projection finale
        output = self.combiner(combined)
        
        return output

class HighCardinalityEncoder(nn.Module):
    """Encodeur spécialisé pour variables à haute cardinalité"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_buckets: int = 10000):
        super().__init__()
        
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, output_dim)
        
        # Réseau pour features continues (si présentes)
        self.continuous_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.combiner = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, categorical_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode features catégorielles à haute cardinalité"""
        
        if "categorical_indices" in categorical_features:
            # Méthode par embedding
            indices = categorical_features["categorical_indices"]
            emb = self.embedding(indices % self.num_buckets)
        else:
            # Initialisation aléatoire
            batch_size = categorical_features.get("batch_size", 1)
            emb = torch.randn(batch_size, self.embedding.embedding_dim)
        
        if "continuous_features" in categorical_features:
            # Combiner avec features continues
            cont_feat = categorical_features["continuous_features"]
            cont_emb = self.continuous_encoder(cont_feat)
            combined = torch.cat([emb, cont_emb], dim=-1)
            output = self.combiner(combined)
        else:
            output = emb
        
        return output

class TemporalEncoder(nn.Module):
    """Encodeur temporel avec Time2Vec et attention temporelle"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # Time2Vec encoding
        self.time2vec = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, output_dim // 2)
        )
        
        # Attention temporelle
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=output_dim // 2,
            num_heads=2,
            batch_first=True
        )
        
        # Projection finale
        self.projection = nn.Linear(output_dim // 2, output_dim)
        
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """Encode features temporelles"""
        
        # Time2Vec encoding
        time_emb = self.time2vec(temporal_features)
        
        # Attention temporelle (si séquence)
        if len(time_emb.shape) == 3:  # [batch, seq_len, features]
            attended, _ = self.temporal_attention(time_emb, time_emb, time_emb)
            # Pooling temporel
            time_encoded = attended.mean(dim=1)
        else:
            time_encoded = time_emb
        
        # Projection finale
        output = self.projection(time_encoded)
        
        return output
