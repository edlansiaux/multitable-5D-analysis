import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from collections import defaultdict

@dataclass
class DimensionMetrics:
    """Métriques pour chaque dimension"""
    volume: Dict[str, float]  # Taille, mémoire, compression
    many_variables: Dict[str, float]  # Nombre, corrélations
    high_cardinality: Dict[str, float]  # Entropie, cardinalité effective
    many_tables: Dict[str, float]  # Densité relationnelle, complexité
    repeated_measurements: Dict[str, float]  # Fréquence, régularité

class DimensionalProfiler:
    """Analyseur automatique des 5 dimensions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics = DimensionMetrics({}, {}, {}, {}, {})
        
    def profile(self, tables: Dict[str, pd.DataFrame], 
                relationships: List[Tuple] = None) -> DimensionMetrics:
        """Analyse complète des dimensions"""
        
        # Dimension 1: Volume
        self._profile_volume(tables)
        
        # Dimension 2: Many Variables
        self._profile_many_variables(tables)
        
        # Dimension 3: High Cardinality
        self._profile_high_cardinality(tables)
        
        # Dimension 4: Many Tables & Relationships
        self._profile_many_tables(tables, relationships)
        
        # Dimension 5: Repeated Measurements
        self._profile_repeated_measurements(tables)
        
        return self.metrics
    
    def _profile_volume(self, tables: Dict[str, pd.DataFrame]):
        """Analyse dimension 1: Volume"""
        total_rows = sum(len(df) for df in tables.values())
        total_memory = sum(df.memory_usage(deep=True).sum() 
                          for df in tables.values())
        
        self.metrics.volume = {
            "total_rows": total_rows,
            "total_memory_mb": total_memory / (1024 ** 2),
            "avg_row_size_kb": (total_memory / total_rows) / 1024 if total_rows > 0 else 0,
            "compression_ratio": self._estimate_compression_ratio(tables),
        }
    
    def _profile_many_variables(self, tables: Dict[str, pd.DataFrame]):
        """Analyse dimension 2: Many Variables"""
        # Analyse de corrélations et redondances
        all_columns = []
        column_stats = {}
        
        for table_name, df in tables.items():
            for col in df.columns:
                col_id = f"{table_name}.{col}"
                all_columns.append(col_id)
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats = {
                        "type": "numeric",
                        "variance": df[col].var(),
                        "skewness": df[col].skew(),
                    }
                else:
                    stats = {"type": "categorical"}
                
                column_stats[col_id] = stats
        
        self.metrics.many_variables = {
            "total_columns": len(all_columns),
            "column_distribution": column_stats,
            "redundancy_score": self._compute_redundancy_score(tables),
        }
    
    def _profile_high_cardinality(self, tables: Dict[str, pd.DataFrame]):
        """Analyse dimension 3: High Cardinality"""
        high_card_metrics = {}
        
        for table_name, df in tables.items():
            for col in df.select_dtypes(include=['object', 'category']).columns:
                unique_count = df[col].nunique()
                total_count = len(df[col])
                cardinality_ratio = unique_count / total_count
                
                # Entropie normalisée
                value_counts = df[col].value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
                norm_entropy = entropy / np.log(unique_count + 1e-10)
                
                high_card_metrics[f"{table_name}.{col}"] = {
                    "unique_count": unique_count,
                    "cardinality_ratio": cardinality_ratio,
                    "entropy": entropy,
                    "normalized_entropy": norm_entropy,
                    "is_high_cardinality": unique_count > 1000,  # Seuil configurable
                }
        
        self.metrics.high_cardinality = high_card_metrics
    
    def _profile_many_tables(self, tables: Dict[str, pd.DataFrame], 
                            relationships: List[Tuple]):
        """Analyse dimension 4: Many Tables & Relationships"""
        n_tables = len(tables)
        n_relationships = len(relationships) if relationships else 0
        
        # Calcul de la densité relationnelle
        max_possible_relationships = n_tables * (n_tables - 1) / 2
        relationship_density = (n_relationships / max_possible_relationships 
                              if max_possible_relationships > 0 else 0)
        
        # Complexité du schéma
        schema_complexity = self._compute_schema_complexity(tables, relationships)
        
        self.metrics.many_tables = {
            "table_count": n_tables,
            "relationship_count": n_relationships,
            "relationship_density": relationship_density,
            "schema_complexity": schema_complexity,
            "avg_columns_per_table": np.mean([len(df.columns) 
                                            for df in tables.values()]),
        }
    
    def _profile_repeated_measurements(self, tables: Dict[str, pd.DataFrame]):
        """Analyse dimension 5: Repeated Measurements"""
        repeated_metrics = {}
        
        for table_name, df in tables.items():
            # Détection des colonnes temporelles
            time_cols = [col for col in df.columns 
                        if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            # Détection des mesures répétées (identifiants avec multiples occurrences)
            potential_id_cols = []
            for col in df.columns:
                if df[col].duplicated().any() and df[col].nunique() < len(df) * 0.9:
                    potential_id_cols.append(col)
            
            repeated_metrics[table_name] = {
                "time_columns": time_cols,
                "potential_id_columns": potential_id_cols,
                "temporal_pattern": self._detect_temporal_pattern(df, time_cols),
            }
        
        self.metrics.repeated_measurements = repeated_metrics
    
    def recommend_pipeline(self) -> Dict[str, Any]:
        """Recommande un pipeline basé sur les métriques analysées"""
        recommendations = {
            "compression_strategy": self._recommend_compression(),
            "embedding_strategy": self._recommend_embedding(),
            "model_architecture": self._recommend_model(),
            "optimization_level": self._recommend_optimization(),
        }
        return recommendations
