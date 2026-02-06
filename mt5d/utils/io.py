import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utilitaire pour le chargement efficace de données multi-tables.
    Supporte CSV et Parquet.
    """
    
    @staticmethod
    def load_table(path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge une table unique depuis un fichier."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Le fichier {path} n'existe pas.")
            
        if path.suffix == '.parquet':
            return pd.read_parquet(path, **kwargs)
        elif path.suffix == '.csv':
            return pd.read_csv(path, **kwargs)
        else:
            raise ValueError(f"Format non supporté : {path.suffix}")

    @staticmethod
    def load_schema_folder(folder_path: str, schema_config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """
        Charge toutes les tables d'un dossier.
        Si schema_config est fourni, ne charge que les tables listées.
        """
        folder = Path(folder_path)
        tables = {}
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.parquet"))
        
        for file_path in files:
            table_name = file_path.stem
            
            # Filtrage si config fournie
            if schema_config and table_name not in schema_config:
                continue
                
            try:
                logger.info(f"Chargement de la table : {table_name}")
                tables[table_name] = DataLoader.load_table(file_path)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {file_path}: {e}")
                
        return tables

    @staticmethod
    def save_results(results: Dict, output_path: str):
        """Sauvegarde les résultats d'analyse (JSON/Pickle)."""
        import pickle
        import json
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Séparation : métriques en JSON (lisible), objets lourds en Pickle
        metrics = results.get('metrics', {})
        if hasattr(metrics, '__dict__'):
            metrics = metrics.__dict__
            
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
            
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(results, f)
