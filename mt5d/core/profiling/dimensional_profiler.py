import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ProfilingMetrics:
    total_volume: int
    variable_counts: Dict[str, int]
    cardinalities: Dict[str, Dict[str, int]]
    repeated_measurements: Dict[str, str] # table -> time_col
    sparsity: float

class DimensionalProfiler:
    """
    Step 0: Predictive Meta-Profiling.
    Analyse les 5 dimensions de complexité avant l'apprentissage.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def profile(self, tables: Dict[str, pd.DataFrame], relationships: List = None) -> ProfilingMetrics:
        print("Analyse des métadonnées (Step 0)...")
        
        total_vol = sum(len(df) for df in tables.values())
        var_counts = {name: len(df.columns) for name, df in tables.items()}
        
        # Détection haute cardinalité (Dim 3)
        cardinalities = {}
        for name, df in tables.items():
            cardinalities[name] = {}
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                n_unique = df[col].nunique()
                if n_unique > 100: # Seuil arbitraire pour "haute"
                    cardinalities[name][col] = n_unique
                    
        # Détection mesures répétées (Dim 5) - Heuristique basique sur noms de colonnes
        time_cols = {}
        time_keywords = ['time', 'date', 'ts', 'created_at']
        for name, df in tables.items():
            for col in df.columns:
                if any(k in col.lower() for k in time_keywords):
                    # Vérifier si c'est une relation one-to-many potentielle (plusieurs entrées par ID)
                    time_cols[name] = col
                    break
                    
        return ProfilingMetrics(
            total_volume=total_vol,
            variable_counts=var_counts,
            cardinalities=cardinalities,
            repeated_measurements=time_cols,
            sparsity=0.0 # À calculer
        )
    
    def recommend_pipeline(self) -> Dict[str, Any]:
        """Recommande des hyperparamètres basés sur le profilage"""
        return {
            "use_pente": True,
            "rht_layers": 3,
            "temporal_dim": 32
        }
