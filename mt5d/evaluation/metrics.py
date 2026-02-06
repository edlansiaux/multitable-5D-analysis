import numpy as np
from typing import Dict

def harmonic_mean(values: List[float]) -> float:
    return len(values) / sum(1.0/v for v in values if v > 0)

def calculate_5d_score(metrics: Dict[str, float]) -> float:
    """
    Definition 11 (Eq 9): 5D Integration Score
    S_5D = HarmonicMean(S1, S2, S3, S4, S5)
    """
    scores = [
        metrics.get('volume_score', 0.5),      # S1
        metrics.get('variable_score', 0.5),    # S2
        metrics.get('cardinality_score', 0.5), # S3
        metrics.get('table_score', 0.5),       # S4
        metrics.get('temporal_score', 0.5)     # S5
    ]
    return harmonic_mean(scores)

def rare_category_recall(predictions, targets, categories, threshold=0.001, k=5):
    """
    B.2: Rare Category Recall@k
    """
    # Implémentation simplifiée
    # Identifier les catégories rares
    unique, counts = np.unique(categories, return_counts=True)
    freqs = counts / len(categories)
    rare_cats = unique[freqs < threshold]
    
    if len(rare_cats) == 0:
        return 0.0
        
    hits = 0
    total_rare = 0
    
    # Simuler le calcul
    # ... Logique de ranking ...
    return 0.0 # Placeholder
