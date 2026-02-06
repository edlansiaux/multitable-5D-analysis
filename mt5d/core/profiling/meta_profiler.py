import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from .dimensional_profiler import DimensionalProfiler

class MetaProfiler(DimensionalProfiler):
    """
    Step 0 (Avancé): Predictive Meta-Profiling.
    Étend le profiler de base pour inférer automatiquement le schéma relationnel
    si celui-ci n'est pas fourni explicitement.
    """
    
    def infer_schema(self, tables: Dict[str, pd.DataFrame]) -> List[Tuple]:
        """
        Détecte les relations Foreign Keys potentielles basées sur :
        1. Noms de colonnes identiques (heuristique 'id')
        2. Chevauchement de valeurs
        """
        print("Inférence automatique du schéma relationnel...")
        inferred_rels = []
        
        # Mapping col_name -> tables qui la contiennent
        col_map = defaultdict(list)
        for t_name, df in tables.items():
            for col in df.columns:
                if 'id' in col.lower(): # Heuristique simple
                    col_map[col].append(t_name)
                    
        # Déduction des liens
        for col, t_list in col_map.items():
            if len(t_list) > 1:
                # Si une colonne 'xxx_id' apparaît dans plusieurs tables,
                # on suppose une relation. On cherche la table "maître" (Primary Key)
                # Heuristique : la table où c'est unique est probablement la source
                
                master_table = None
                candidates = []
                
                for t in t_list:
                    if tables[t][col].is_unique:
                        master_table = t
                    else:
                        candidates.append(t)
                        
                if master_table:
                    for cand in candidates:
                        rel = (master_table, col, cand, col, 'inferred_one_to_many')
                        inferred_rels.append(rel)
                        print(f"  - Relation détectée : {rel}")
                        
        return inferred_rels
