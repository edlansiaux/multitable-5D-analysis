import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalDiscoveryLoss(nn.Module):
    """
    Definition 7 (Eq 4): Relational Discovery Loss
    L_rel = alpha * L_task + beta * L_sparse + gamma * L_semantic
    """
    def __init__(self, alpha=1.0, beta=0.01, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.task_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, target, adj_matrix, semantic_sim_matrix):
        # 1. Task Loss
        l_task = self.task_loss(pred, target)
        
        # 2. Sparsity Loss (L1 sur la matrice d'adjacence apprise)
        l_sparse = torch.mean(torch.abs(adj_matrix))
        
        # 3. Semantic Loss (Cohérence sémantique)
        # On veut que les arêtes fortes (adj_matrix) correspondent à une similarité sémantique élevée
        # Loss = sum(weight * (1 - semantic_sim))
        l_semantic = torch.mean(adj_matrix * (1.0 - semantic_sim_matrix))
        
        return self.alpha * l_task + self.beta * l_sparse + self.gamma * l_semantic

class ContrastiveRelationalLoss(nn.Module):
    """
    Definition 9 (Eq 6): Relational-Temporal Contrastive Loss
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j, is_related):
        """
        z_i, z_j: Embeddings PentE de deux batchs d'entités
        is_related: Masque binaire [batch, batch] indiquant si les entités sont liées
        """
        # Similarité Cosinus
        sim = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2) / self.temperature
        
        # Pour Contrastive Learning, on veut maximiser sim si is_related=1, minimiser sinon.
        # Implémentation simplifiée type InfoNCE
        exp_sim = torch.exp(sim)
        
        # Numérateur: exp(sim) pour les paires positives
        pos_mask = is_related.bool()
        numerator = exp_sim * pos_mask.float()
        
        # Dénominateur: somme des exp(sim) pour tous
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Log probability
        log_prob = torch.log(numerator.sum(dim=1) / denominator + 1e-8)
        
        return -log_prob.mean()
