from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import pandas as pd

class BaseDataset(ABC):
    """
    Classe abstraite pour les datasets multi-tables (Section 7.1).
    Tout nouveau dataset doit hériter de cette classe.
    """
    
    @abstractmethod
    def load(self) -> Tuple[Dict[str, pd.DataFrame], List[Tuple]]:
        """
        Charge les données et retourne :
        1. Un dictionnaire de DataFrames {nom_table: df}
        2. Une liste de relations [(table_src, col_src, table_tgt, col_tgt, type)]
        """
        pass
    
    @abstractmethod
    def get_temporal_columns(self) -> Dict[str, str]:
        """
        Retourne le mapping {nom_table: nom_colonne_temporelle}
        pour la gestion de la dimension 5 (Mesures répétées).
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retourne des métadonnées descriptives (optionnel)."""
        return {}
