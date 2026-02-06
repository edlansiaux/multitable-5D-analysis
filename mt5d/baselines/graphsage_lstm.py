import torch
import torch.nn as nn
import dgl.nn as dglnn

class GraphSAGELSTMBaseline(nn.Module):
    """
    Baseline: GraphSAGE + LSTM
    Approche hybride classique pour graphes dynamiques.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Composante Structurelle
        self.sage = dglnn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean')
        
        # Composante Temporelle (séquentielle)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.classifier = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, g, node_feats, history_feats=None):
        # 1. Agrégation structurelle (GraphSAGE)
        h_struct = self.sage(g, node_feats)
        
        # 2. Agrégation temporelle (LSTM)
        # Si on a un historique, on passe dans le LSTM
        if history_feats is not None:
            # history_feats: [batch, seq_len, hidden_dim]
            lstm_out, _ = self.lstm(history_feats)
            h_temp = lstm_out[:, -1, :] # Dernière étape
            h_combined = h_struct + h_temp
        else:
            h_combined = h_struct
            
        return self.classifier(h_combined)
