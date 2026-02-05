"""
Fonctions de perte pour l'apprentissage relationnel multitable
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class RelationalContrastiveLoss(nn.Module):
    """Perte contrastive relationnelle"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
    
    def forward(self, embeddings: torch.Tensor, 
                relations: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        embeddings: [batch_size, embedding_dim]
        relations: [batch_size, batch_size] matrice d'adjacence
        labels: [batch_size] labels pour supervised contrastive
        """
        
        batch_size = embeddings.size(0)
        
        # Similarités cosinus
        sim_matrix = self.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0)
        ) / self.temperature
        
        # Masques pour positive/negative pairs
        if labels is not None:
            # Supervised: positives = même label
            label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            positive_mask = label_matrix - torch.eye(batch_size, device=embeddings.device)
        else:
            # Self-supervised: positives = relations dans le graphe
            positive_mask = relations - torch.eye(batch_size, device=embeddings.device)
        
        negative_mask = 1 - positive_mask - torch.eye(batch_size, device=embeddings.device)
        
        # Perte pour les positive pairs
        pos_sim = sim_matrix * positive_mask
        pos_loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(sim_matrix) * negative_mask).sum(dim=1, keepdim=True) + 1e-8
        )
        pos_loss = (pos_loss * positive_mask).sum() / (positive_mask.sum() + 1e-8)
        
        # Perte pour les negative pairs (margin-based)
        neg_sim = sim_matrix * negative_mask
        neg_loss = F.relu(neg_sim + self.margin).mean()
        
        return pos_loss + neg_loss

class TemporalConsistencyLoss(nn.Module):
    """Perte de consistance temporelle"""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.01):
        super().__init__()
        self.alpha = alpha  # Poids pour la consistance
        self.beta = beta    # Poids pour la régularisation
        
    def forward(self, embeddings: torch.Tensor,
                timestamps: torch.Tensor,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [batch_size, seq_len, embedding_dim]
        timestamps: [batch_size, seq_len]
        predictions: [batch_size, seq_len, output_dim]
        targets: [batch_size, seq_len, output_dim]
        """
        
        # Perte de reconstruction
        reconstruction_loss = F.mse_loss(predictions, targets)
        
        # Perte de consistance temporelle
        time_diff = timestamps[:, 1:] - timestamps[:, :-1]
        time_diff = time_diff.unsqueeze(-1)
        
        emb_diff = embeddings[:, 1:] - embeddings[:, :-1]
        
        # Normaliser par différence temporelle
        time_consistency_loss = (emb_diff / (time_diff + 1e-8)).pow(2).mean()
        
        # Régularisation pour smoothness
        smoothness_loss = emb_diff.pow(2).mean()
        
        total_loss = reconstruction_loss + \
                    self.alpha * time_consistency_loss + \
                    self.beta * smoothness_loss
        
        return total_loss

class HighCardinalityLoss(nn.Module):
    """Perte spécialisée pour variables à haute cardinalité"""
    
    def __init__(self, num_classes: int, label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logits: torch.Tensor, 
                targets: torch.Tensor,
                class_weights: Optional[torch.Tensor] = None,
                rare_class_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]
        class_weights: [num_classes]
        rare_class_indices: indices des classes rares
        """
        
        # Perte cross-entropy de base
        if class_weights is not None:
            loss = F.cross_entropy(logits, targets, weight=class_weights)
        else:
            loss = self.cross_entropy(logits, targets)
        
        # Pénalité supplémentaire pour classes rares
        if rare_class_indices:
            rare_mask = torch.zeros_like(logits)
            rare_mask[:, rare_class_indices] = 1
            
            # Focus sur les prédictions incorrectes pour classes rares
            preds = torch.argmax(logits, dim=1)
            incorrect = (preds != targets).float().unsqueeze(1)
            
            rare_penalty = (logits * rare_mask * incorrect).abs().mean()
            loss = loss + 0.5 * rare_penalty
        
        return loss

class MultiTaskLoss(nn.Module):
    """Perte multi-tâche avec pondération automatique"""
    
    def __init__(self, num_tasks: int, 
                loss_types: List[str],
                initial_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.loss_types = loss_types
        
        # Pondérations apprenables
        if initial_weights:
            self.task_weights = nn.Parameter(torch.tensor(initial_weights))
        else:
            self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
        # Initialiser les fonctions de perte
        self.loss_functions = []
        for loss_type in loss_types:
            if loss_type == 'classification':
                self.loss_functions.append(nn.CrossEntropyLoss())
            elif loss_type == 'regression':
                self.loss_functions.append(nn.MSELoss())
            elif loss_type == 'binary':
                self.loss_functions.append(nn.BCEWithLogitsLoss())
            else:
                raise ValueError(f"Type de perte non supporté: {loss_type}")
    
    def forward(self, predictions: List[torch.Tensor], 
                targets: List[torch.Tensor]) -> torch.Tensor:
        """
        predictions: liste de tenseurs [batch_size, ...]
        targets: liste de tenseurs [batch_size, ...]
        """
        
        losses = []
        for i, (pred, target, loss_fn) in enumerate(zip(predictions, targets, self.loss_functions)):
            loss = loss_fn(pred, target)
            weighted_loss = torch.exp(-self.task_weights[i]) * loss + self.task_weights[i]
            losses.append(weighted_loss)
        
        total_loss = sum(losses)
        return total_loss

class GraphRegularizationLoss(nn.Module):
    """Perte de régularisation pour graphes"""
    
    def __init__(self, lambda_smooth: float = 0.1, 
                lambda_ortho: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_ortho = lambda_ortho
    
    def forward(self, embeddings: torch.Tensor, 
                adjacency: torch.Tensor,
                laplacian: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        embeddings: [num_nodes, embedding_dim]
        adjacency: [num_nodes, num_nodes] matrice d'adjacence
        laplacian: [num_nodes, num_nodes] matrice laplacienne
        """
        
        loss = 0
        
        # Régularisation de smoothness (graph Laplacian)
        if laplacian is not None:
            smoothness = torch.trace(embeddings.T @ laplacian @ embeddings)
            loss += self.lambda_smooth * smoothness
        
        # Régularisation d'orthogonalité
        if embeddings.size(1) > 1:
            correlation = embeddings.T @ embeddings
            identity = torch.eye(embeddings.size(1), device=embeddings.device)
            ortho_loss = F.mse_loss(correlation, identity, reduction='mean')
            loss += self.lambda_ortho * ortho_loss
        
        return loss

class PentELoss(nn.Module):
    """Perte combinée pour l'embedding pentadimensionnel"""
    
    def __init__(self, 
                 alpha: float = 1.0,      # Reconstruction
                 beta: float = 0.5,       # Relationnelle
                 gamma: float = 0.3,      # Temporelle
                 delta: float = 0.2,      # Catégorielle
                 epsilon: float = 0.1):   # Volume
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        
        # Sous-pertes
        self.reconstruction_loss = nn.MSELoss()
        self.relational_loss = RelationalContrastiveLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.categorical_loss = HighCardinalityLoss(num_classes=1000)
        self.volume_loss = nn.L1Loss()
    
    def forward(self, 
                reconstructed: torch.Tensor,
                original: torch.Tensor,
                embeddings: torch.Tensor,
                relations: torch.Tensor,
                temporal_embeddings: torch.Tensor,
                timestamps: torch.Tensor,
                categorical_logits: torch.Tensor,
                categorical_targets: torch.Tensor,
                volume_pred: torch.Tensor,
                volume_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calcule la perte totale PentE"""
        
        losses = {}
        
        # 1. Reconstruction loss
        losses['reconstruction'] = self.alpha * self.reconstruction_loss(
            reconstructed, original
        )
        
        # 2. Relational loss
        losses['relational'] = self.beta * self.relational_loss(
            embeddings, relations
        )
        
        # 3. Temporal loss
        losses['temporal'] = self.gamma * self.temporal_loss(
            temporal_embeddings, timestamps,
            temporal_embeddings, temporal_embeddings  # Auto-reconstruction
        )
        
        # 4. Categorical loss
        losses['categorical'] = self.delta * self.categorical_loss(
            categorical_logits, categorical_targets
        )
        
        # 5. Volume loss
        losses['volume'] = self.epsilon * self.volume_loss(
            volume_pred, volume_target
        )
        
        # Perte totale
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses

class AdaptiveWeightedLoss(nn.Module):
    """Perte avec pondération adaptative basée sur la difficulté"""
    
    def __init__(self, base_loss_fn: nn.Module = nn.MSELoss(),
                adaptation_rate: float = 0.1):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.adaptation_rate = adaptation_rate
        self.register_buffer('sample_weights', None)
    
    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                difficulty_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        predictions: [batch_size, ...]
        targets: [batch_size, ...]
        difficulty_scores: [batch_size] scores de difficulté (0=easy, 1=hard)
        """
        
        # Perte par échantillon
        per_sample_loss = self.base_loss_fn(
            predictions, targets, reduction='none'
        )
        
        # Aplatir pour avoir un scalaire par échantillon
        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)
        
        # Pondération adaptative
        if difficulty_scores is not None:
            # Poids inversement proportionnels à la difficulté
            weights = 1.0 / (difficulty_scores + 1e-8)
            weights = weights / weights.sum() * weights.size(0)  # Normaliser
        else:
            # Apprentissage automatique des poids
            if self.sample_weights is None:
                self.sample_weights = torch.ones_like(per_sample_loss)
            
            # Mettre à jour les poids basé sur la perte
            with torch.no_grad():
                loss_ratio = per_sample_loss / (per_sample_loss.mean() + 1e-8)
                self.sample_weights = (1 - self.adaptation_rate) * self.sample_weights + \
                                     self.adaptation_rate * loss_ratio
            
            weights = self.sample_weights
        
        # Perte pondérée
        weighted_loss = (per_sample_loss * weights).mean()
        
        return weighted_loss

def create_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory pour créer des fonctions de perte"""
    
    loss_registry = {
        'mse': nn.MSELoss,
        'l1': nn.L1Loss,
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCEWithLogitsLoss,
        'relational_contrastive': RelationalContrastiveLoss,
        'temporal_consistency': TemporalConsistencyLoss,
        'high_cardinality': HighCardinalityLoss,
        'multi_task': MultiTaskLoss,
        'graph_regularization': GraphRegularizationLoss,
        'pente': PentELoss,
        'adaptive_weighted': AdaptiveWeightedLoss
    }
    
    if loss_name not in loss_registry:
        raise ValueError(f"Fonction de perte non supportée: {loss_name}")
    
    return loss_registry[loss_name](**kwargs)

class LossTracker:
    """Tracker pour suivre l'évolution des pertes"""
    
    def __init__(self):
        self.history = {}
        self.current_epoch = {}
    
    def update(self, loss_name: str, value: float):
        """Met à jour la perte pour l'époque actuelle"""
        if loss_name not in self.current_epoch:
            self.current_epoch[loss_name] = []
        self.current_epoch[loss_name].append(value)
    
    def end_epoch(self):
        """Termine l'époque et enregistre les moyennes"""
        for loss_name, values in self.current_epoch.items():
            if loss_name not in self.history:
                self.history[loss_name] = []
            
            if values:
                self.history[loss_name].append(np.mean(values))
        
        self.current_epoch.clear()
    
    def get_history(self) -> Dict[str, List[float]]:
        """Retourne l'historique des pertes"""
        return self.history
    
    def plot_history(self, figsize: tuple = (12, 8)):
        """Visualise l'historique des pertes"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Toutes les pertes
        ax1 = axes[0]
        for loss_name, values in self.history.items():
            ax1.plot(values, label=loss_name, linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Évolution des Pertes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Perte totale seulement
        ax2 = axes[1]
        if 'total' in self.history:
            ax2.plot(self.history['total'], linewidth=3, color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Totale')
            ax2.set_title('Perte Totale')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
