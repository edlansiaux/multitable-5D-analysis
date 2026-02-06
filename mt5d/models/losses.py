"""
Fonctions de perte alignées avec le manuscrit (Sections 4.2.4 et 5.4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalDiscoveryLoss(nn.Module):
    """
    Definition 7 (Eq. 4): Relational Discovery Loss
    L_rel = alpha * L_task + beta * L_sparse + gamma * L_semantic
    """
    def __init__(self, alpha=1.0, beta=0.01, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.task_loss = nn.CrossEntropyLoss() # Par défaut, adaptable

    def forward(self, pred, target, adj_matrix, node_embeddings):
        # 1. Task Loss (L_task)
        l_task = self.task_loss(pred, target)
        
        # 2. Sparsity Loss (L_sparse) - Pénalise la densité du graphe appris
        # On suppose que adj_matrix contient des poids d'attention ou probabilités
        l_sparse = torch.mean(torch.abs(adj_matrix))
        
        # 3. Semantic Consistency (L_semantic)
        # Les nœuds connectés doivent être sémantiquement proches
        # On calcule la distance moyenne pondérée par l'adjacence
        if adj_matrix.is_sparse:
            adj_dense = adj_matrix.to_dense()
        else:
            adj_dense = adj_matrix
            
        # Similarité cosinus entre tous les nœuds (coûteux, simplifié ici pour l'exemple)
        # Version optimisée : sampling d'arêtes
        sim_matrix = F.cosine_similarity(node_embeddings.unsqueeze(1), node_embeddings.unsqueeze(0), dim=2)
        # On veut maximiser la similarité là où adj est fort -> minimiser (1 - sim) * adj
        l_semantic = torch.mean(adj_dense * (1 - sim_matrix))

        return self.alpha * l_task + self.beta * l_sparse + self.gamma * l_semantic

class RelationalTemporalContrastiveLoss(nn.Module):
    """
    Definition 9 (Eq. 6): Relational-Temporal Contrastive Loss
    L_CRT = alpha * L_rel + beta * L_temp + gamma * L_sem
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, adj, temporal_sim):
        """
        z: Embeddings [N, D]
        adj: Matrice d'adjacence binaire ou pondérée [N, N] (Relations)
        temporal_sim: Matrice de proximité temporelle [N, N]
        """
        # Similarité dans l'espace latent
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        exp_sim = torch.exp(sim)
        
        # Masque pour exclure self-loops
        mask = torch.eye(z.shape[0], device=z.device).bool()
        exp_sim.masked_fill_(mask, 0)
        
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # L_rel: Maximiser log-likelihood pour les voisins relationnels
        log_prob = sim - torch.log(denominator + 1e-8)
        # On ne garde que les paires positives (adj > 0)
        l_rel = -(adj * log_prob).sum(dim=1) / (adj.sum(dim=1) + 1e-8)
        
        # L_temp: Idem pour voisins temporels
        l_temp = -(temporal_sim * log_prob).sum(dim=1) / (temporal_sim.sum(dim=1) + 1e-8)
        
        # L_sem: Encourager la cohérence intrinsèque (ex: via augmentation de données, simplifié ici)
        l_sem = 0.0 # Placeholder pour terme sémantique pur
        
        return l_rel.mean() + l_temp.mean() + l_sem
