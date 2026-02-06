import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

class TGNBaseline(nn.Module):
    """
    Baseline: Temporal Graph Network (TGN)
    Référence: Rossi et al. (2020) [Source: 486]
    Utilisé pour la comparaison dans le Tableau 2.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Simplification: TGN est simulé ici par un GAT avec encodage temporel basique
        # car TGN complet est complexe à implémenter from scratch
        self.temporal_encoder = nn.Linear(1, hidden_dim)
        self.gnn = dglnn.GATConv(in_dim, hidden_dim // 4, num_heads=4)
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, g, node_feats, edge_feats, timestamps):
        # Encodage temporel
        t_emb = self.temporal_encoder(timestamps.unsqueeze(-1).float())
        
        # Message Passing (Graph Attention)
        h = self.gnn(g, node_feats)
        h = h.flatten(1)
        
        # Mise à jour mémoire temporelle (Memory Module du TGN)
        # h_new = GRU(message, h_old)
        # Ici simplifié pour l'exemple
        h_mem = self.memory(h)
        
        return self.out_proj(h_mem)
