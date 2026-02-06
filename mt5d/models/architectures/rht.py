import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from typing import Dict, List, Optional, Tuple
from ..embeddings.pente import PentE

class RelationalHypergraphTransformer(nn.Module):
    """
    Relational Hypergraph Transformer (RHT)
    Correspondance Manuscrit : Section 4
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_pente: bool = True):
        super().__init__()
        
        self.use_pente = use_pente
        self.hidden_dim = hidden_dim
        
        if use_pente:
            self.pente = PentE(
                node_dim=input_dim,
                output_dim=hidden_dim
            )
            # La dimension d'entrée des couches suivantes est la sortie de PentE
            curr_dim = hidden_dim 
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            curr_dim = hidden_dim
        
        # Module 4: Differentiable Relational Discovery (Section 4.2.4)
        self.graph_rewiring = GraphRewiringModule(hidden_dim)
        
        # Couches de message passing sur hypergraphe
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = RelationalHypergraphLayer(
                in_dim=curr_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.layers.append(layer)
            curr_dim = hidden_dim
        
        # Normalisation et projection finale
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, 
                g: dgl.DGLGraph,
                node_features: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                temporal_info: Optional[torch.Tensor] = None,
                categorical_info: Optional[Dict] = None):
        """
        Forward pass du RHT
        """
        
        # Step 2: Unified 5D Embedding (PentE)
        if self.use_pente:
            h = self.pente(
                node_features=node_features,
                relation_features=edge_features if edge_features is not None else torch.zeros(g.num_edges(), 64, device=g.device),
                temporal_features=temporal_info if temporal_info is not None else torch.zeros(g.num_nodes(), 32, device=g.device),
                categorical_features=categorical_info if categorical_info else {},
                volume_features=torch.ones(g.num_nodes(), 16, device=g.device) # Placeholder si non fourni
            )
        else:
            h = self.input_proj(node_features)
        
        # Step 4: Dynamic Graph Rewiring
        # On ajuste la structure du graphe basée sur les embeddings actuels
        adj_weight = self.graph_rewiring(g, h)
        
        # Message passing sur hypergraphe avec Attention Clairsemée
        for layer in self.layers:
            h = layer(g, h, adj_weight)
        
        # Normalisation et sortie
        h = self.norm(h)
        output = self.output_proj(h)
        
        return output

class HyperEdgeAttention(nn.Module):
    """
    Module 6.1: Sparse Relational Attention
    Implémentation de l'Listing 1 du manuscrit.
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        # Pour simplifier l'implémentation DGL, on utilise le message passing 
        # pour simuler l'attention clairsemée définie par la topologie du graphe
        
        with g.local_scope():
            Q = self.q_proj(h).view(-1, self.num_heads, self.head_dim)
            K = self.k_proj(h).view(-1, self.num_heads, self.head_dim)
            V = self.v_proj(h).view(-1, self.num_heads, self.head_dim)
            
            g.ndata['Q'] = Q
            g.ndata['K'] = K
            g.ndata['V'] = V
            
            # Attention calculée uniquement sur les arêtes existantes (Sparse)
            # Score = (Q * K) / sqrt(d)
            g.apply_edges(dgl.function.u_dot_v('Q', 'K', 'score'))
            g.edata['score'] = g.edata['score'] * self.scale
            
            # Softmax sur les voisins (sparse softmax)
            g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['score'])
            g.edata['a'] = self.dropout(g.edata['a'])
            
            # Agrégation pondérée des valeurs
            g.update_all(dgl.function.u_mul_e('V', 'a', 'm'),
                         dgl.function.sum('m', 'h_out'))
            
            h_out = g.ndata['h_out'].view(-1, self.num_heads * self.head_dim)
            return self.out_proj(h_out)

class RelationalHypergraphLayer(nn.Module):
    """
    Couche de base combinant Attention et Message Passing
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        # Attention relationnelle (Section 6.1)
        self.attention = HyperEdgeAttention(in_dim, out_dim, num_heads, dropout)
        
        # Message passing temporel-relationnel (Section 6.3)
        # Ici simplifié via DGL GraphConv, idéalement customisé pour Eq 8
        self.graph_conv = dglnn.GraphConv(out_dim, out_dim)
        
        # FFN standard
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim)
        )
        
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm3 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        # 1. Attention Mechanism
        h_attn = self.attention(g, h)
        h = self.norm1(h + self.dropout(h_attn))
        
        # 2. Structure Propagation (Message Passing)
        h_struct = self.graph_conv(g, h, edge_weight=edge_weight)
        h = self.norm2(h + self.dropout(h_struct))
        
        # 3. Feed Forward
        h_ffn = self.ffn(h)
        h = self.norm3(h + self.dropout(h_ffn))
        
        return h

class GraphRewiringModule(nn.Module):
    """
    Module 4 & 5.5: Differentiable Relational Discovery & Graph Rewiring
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Prédicteur de lien : concaténation des features de deux nœuds -> probabilité
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        """
        Calcule les poids d'attention pour les arêtes existantes 
        et pourrait en proposer de nouvelles (version simplifiée ici : re-weighting)
        """
        with g.local_scope():
            g.ndata['h'] = h
            # Pour chaque arête existante, recalculer son importance basée sur le contexte actuel
            g.apply_edges(self._calc_edge_score)
            return g.edata['score'].squeeze()
            
    def _calc_edge_score(self, edges):
        # Concaténation source + destination
        h_cat = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'score': self.link_predictor(h_cat)}
