import numpy as np
import torch
from scipy.spatial.distance import cosine

class RelationalDriftDetector:
    """
    Step 7: Monitoring & Drift Detection.
    Détecte les changements dans la structure relationnelle ou les distributions.
    """
    
    def __init__(self, reference_embeddings=None):
        self.ref_emb = reference_embeddings
        self.threshold = 0.1
        
    def set_reference(self, embeddings: torch.Tensor):
        # Stocke les statistiques des embeddings de référence (validation set)
        self.ref_mean = torch.mean(embeddings, dim=0)
        self.ref_std = torch.std(embeddings, dim=0)
        
    def detect_drift(self, new_embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Calcule la dérive entre les embeddings de référence et les nouveaux.
        """
        if not hasattr(self, 'ref_mean'):
            return {"drift_score": 0.0, "status": "no_reference"}
            
        curr_mean = torch.mean(new_embeddings, dim=0)
        
        # Distance cosinus entre les centroïdes des embeddings
        # Une dérive sémantique ou relationnelle déplacera le centroïde dans l'espace PentE
        drift_magnitude = 1.0 - torch.nn.functional.cosine_similarity(
            self.ref_mean.unsqueeze(0), curr_mean.unsqueeze(0)
        ).item()
        
        is_drifting = drift_magnitude > self.threshold
        
        return {
            "drift_score": drift_magnitude,
            "is_drifting": is_drifting,
            "status": "warning" if is_drifting else "stable"
        }
