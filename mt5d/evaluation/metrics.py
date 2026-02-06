"""
Métriques d'évaluation conformes au manuscrit (Section 7.2)
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import hmean

class MT5DMetrics:
    def __init__(self):
        self.results = {}

    def compute_5d_integration_score(self, dim_scores: Dict[str, float]) -> float:
        """
        Definition 11 (Eq. 9): 5D Integration Score
        Harmonic Mean of normalized scores for each dimension.
        """
        scores = [
            dim_scores.get('volume', 0.5),
            dim_scores.get('variables', 0.5),
            dim_scores.get('cardinality', 0.5),
            dim_scores.get('tables', 0.5),
            dim_scores.get('temporal', 0.5)
        ]
        
        # Éviter division par zéro
        scores = [max(s, 1e-6) for s in scores]
        
        # Moyenne harmonique
        s_5d = hmean(scores)
        return s_5d

    def rare_category_recall_at_k(self, y_true, y_pred_proba, k=10, threshold=0.001):
        """
        Definition (Eq. 13): Rare Category Recall@k
        RCR@k pour les catégories avec fréquence < 0.1%
        """
        # 1. Identifier les catégories rares
        unique, counts = np.unique(y_true, return_counts=True)
        freqs = counts / len(y_true)
        rare_cats = unique[freqs < threshold]
        
        if len(rare_cats) == 0:
            return 0.0
            
        # 2. Calculer le Recall@k pour ces catégories
        hits = 0
        total_rare = 0
        
        # Top-k predictions indices
        if y_pred_proba.ndim > 1:
            top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        else:
            return 0.0
            
        for i, true_label in enumerate(y_true):
            if true_label in rare_cats:
                total_rare += 1
                if true_label in top_k_preds[i]:
                    hits += 1
                    
        return hits / total_rare if total_rare > 0 else 0.0

    def generate_report(self):
        return pd.DataFrame([self.results])
