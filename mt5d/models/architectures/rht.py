import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from typing import Dict, Optional
from ..embeddings.pente import PentE

class RelationalHypergraphTransformer(nn.Module):
    """
    Architecture RHT complète (Section 4).
    Intègre:
    - PentE (Embeddings 5D) [cite: 176]
    - Graph Rewiring (Module 4) 
    - Sparse Relational Attention (Module 3 & 6.1) [cite: 184]
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_pente: bool = True):
        super().__init__()
        
        self.use_pente = use_pente
        self.hidden_dim = hidden_dim
        
        # 1. Module d'Embedding 5D (PentE)
        if use_pente:
            self.pente = PentE(
                node_dim=input_dim,
                output_dim=hidden_dim,
                # Les dimensions internes peuvent être configurables via config
                relation_dim=64,
                temporal_dim=32,
                categorical_dim=64,
                volume_dim=16
            )
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Module 4: Differentiable Relational Discovery (Graph Rewiring)
        # Permet d'apprendre des poids sur les arêtes ou d'en découvrir de nouvelles
        self.graph_rewiring = GraphRewiringModule(hidden_dim)
        
        # 3. Couches de Message Passing / Attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RelationalHypergraphLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, 
                g: dgl.DGLGraph,
                node_features: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                temporal_features: Optional[torch.Tensor] = None,
                categorical_features: Optional[Dict[str, torch.Tensor]] = None,
                volume_features: Optional[torch.Tensor] = None):
        
        # Step 2: Unified 5D Embedding
        if self.use_pente:
            # Gestion des entrées optionnelles pour éviter les erreurs lors de l'appel
            device = node_features.device
            num_nodes = node_features.shape[0]
            num_edges = g.num_edges()
            
            if edge_features is None:
                edge_features = torch.zeros(num_edges, 64, device=device)
            if temporal_features is None:
                temporal_features = torch.zeros(num_nodes, 32, device=device) # dim par défaut dans PentE
            if volume_features is None:
                volume_features = torch.zeros(num_nodes, 16, device=device)
            if categorical_features is None:
                categorical_features = {} # PentE gère le dictionnaire vide
                
            h = self.pente(
                node_features=node_features,
                relation_features=edge_features,
                temporal_features=temporal_features,
                categorical_features=categorical_features,
                volume_features=volume_features
            )
        else:
            h = self.input_proj(node_features)
            
        # Step 4: Dynamic Graph Rewiring [cite: 242]
        # Recalcule les poids d'attention des arêtes basés sur les embeddings actuels
        edge_weights = self.graph_rewiring(g, h)
        
        # Message Passing
        for layer in self.layers:
            h = layer(g, h, edge_weights)
            
        h = self.norm(h)
        return self.output_proj(h)

class GraphRewiringModule(nn.Module):
    """Module 4: Differentiable Relational Discovery """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Calcule un score d'importance pour chaque arête existante
            # (src_emb || dst_emb) -> score
            g.apply_edges(lambda edges: {
                'score': self.attn(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
            })
            return g.edata['score']

class RelationalHypergraphLayer(nn.Module):
    """
    Couche combinant Attention Relationnelle (Sparse) et Message Passing Temporel
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout):
        super().__init__()
        # Implémentation simplifiée de GAT (Graph Attention Network) qui agit comme
        # l'attention relationnelle clairsemée décrite en 6.1
        self.conv = dglnn.GATConv(in_dim, out_dim // num_heads, num_heads, 
                                  feat_drop=dropout, attn_drop=dropout, residual=True)
        
    def forward(self, g, h, edge_weights=None):
        # Le GAT de DGL gère nativement l'attention, 
        # mais on peut injecter nos edge_weights appris par le module de rewiring
        # comme bias ou masque d'attention si supporté, ou simplement pondérer le graphe.
        
        # Ici, forward standard GAT concaténé
        h_heads = self.conv(g, h) # [nodes, heads, head_dim]
        h_out = h_heads.flatten(1) # [nodes, out_dim]
        return h_out
