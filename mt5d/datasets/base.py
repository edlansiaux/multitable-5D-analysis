"""
Classe de base pour les datasets MT5D
"""
import abc
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

class DatasetType(Enum):
    """Types de datasets supportés"""
    MEDICAL = "medical"
    ECOMMERCE = "ecommerce"
    FINANCIAL = "financial"
    SOCIAL = "social"
    IOT = "iot"
    SYNTHETIC = "synthetic"

@dataclass
class DatasetMetadata:
    """Métadonnées d'un dataset"""
    name: str
    type: DatasetType
    description: str
    source: str
    license: str
    num_tables: int
    total_rows: int
    total_columns: int
    has_temporal_data: bool
    has_relationships: bool
    download_url: Optional[str] = None
    citation: Optional[str] = None
    version: str = "1.0.0"

class MT5DDataset(abc.ABC):
    """Classe abstraite pour les datasets MT5D"""
    
    def __init__(self, data_dir: str, download: bool = False):
        self.data_dir = Path(data_dir)
        self.metadata = None
        self.tables = {}
        self.relationships = []
        
        if download:
            self.download()
    
    @abc.abstractmethod
    def download(self):
        """Télécharge le dataset si nécessaire"""
        pass
    
    @abc.abstractmethod
    def load(self) -> Dict[str, pd.DataFrame]:
        """Charge les tables du dataset"""
        pass
    
    @abc.abstractmethod
    def get_relationships(self) -> List[Tuple]:
        """Retourne les relations entre tables"""
        pass
    
    @abc.abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Retourne les métadonnées du dataset"""
        pass
    
    def validate(self) -> bool:
        """Valide l'intégrité du dataset"""
        # Vérifier que toutes les tables existent
        if not self.tables:
            self.load()
        
        # Vérifier les relations
        relationships = self.get_relationships()
        
        errors = []
        for rel in relationships:
            if len(rel) < 4:
                errors.append(f"Relation invalide: {rel}")
                continue
            
            src_table, src_col, tgt_table, tgt_col = rel[:4]
            
            if src_table not in self.tables:
                errors.append(f"Table source non trouvée: {src_table}")
            
            if tgt_table not in self.tables:
                errors.append(f"Table cible non trouvée: {tgt_table}")
            
            if src_table in self.tables and src_col not in self.tables[src_table].columns:
                errors.append(f"Colonne {src_col} non trouvée dans {src_table}")
            
            if tgt_table in self.tables and tgt_col not in self.tables[tgt_table].columns:
                errors.append(f"Colonne {tgt_col} non trouvée dans {tgt_table}")
        
        if errors:
            print("Erreurs de validation:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_table_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les statistiques de chaque table"""
        stats = {}
        
        for table_name, df in self.tables.items():
            stats[table_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'temporal_columns': len(df.select_dtypes(include=['datetime']).columns),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
        
        return stats
    
    def save(self, output_dir: str, format: str = 'parquet'):
        """Sauvegarde le dataset dans un format spécifique"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les tables
        for table_name, df in self.tables.items():
            if format == 'parquet':
                df.to_parquet(output_dir / f"{table_name}.parquet")
            elif format == 'csv':
                df.to_csv(output_dir / f"{table_name}.csv", index=False)
            elif format == 'feather':
                df.to_feather(output_dir / f"{table_name}.feather")
        
        # Sauvegarder les relations
        relationships = self.get_relationships()
        with open(output_dir / 'relationships.json', 'w') as f:
            json.dump([
                {
                    'source_table': src,
                    'source_column': src_col,
                    'target_table': tgt,
                    'target_column': tgt_col,
                    'relation_type': rel_type
                }
                for src, src_col, tgt, tgt_col, rel_type in relationships
            ], f, indent=2)
        
        # Sauvegarder les métadonnées
        metadata = self.get_metadata()
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump({
                'name': metadata.name,
                'type': metadata.type.value,
                'description': metadata.description,
                'source': metadata.source,
                'license': metadata.license,
                'num_tables': metadata.num_tables,
                'total_rows': metadata.total_rows,
                'total_columns': metadata.total_columns,
                'has_temporal_data': metadata.has_temporal_data,
                'has_relationships': metadata.has_relationships,
                'citation': metadata.citation,
                'version': metadata.version
            }, f, indent=2)
        
        print(f"Dataset sauvegardé dans: {output_dir}")
    
    def create_sample(self, sample_size: float = 0.1, random_state: int = 42):
        """Crée un échantillon du dataset"""
        sampled_tables = {}
        
        for table_name, df in self.tables.items():
            # Pour les tables de référence (peu de lignes), garder tout
            if len(df) < 1000:
                sampled_tables[table_name] = df.copy()
            else:
                sampled_tables[table_name] = df.sample(
                    frac=sample_size, 
                    random_state=random_state
                )
        
        # Mettre à jour les relations si nécessaire
        sampled_relationships = self.get_relationships()
        
        return sampled_tables, sampled_relationships

class DatasetRegistry:
    """Registre des datasets disponibles"""
    
    _datasets = {}
    
    @classmethod
    def register(cls, name: str, dataset_class):
        """Enregistre un nouveau dataset"""
        cls._datasets[name] = dataset_class
    
    @classmethod
    def get_dataset(cls, name: str, data_dir: str, **kwargs):
        """Retourne une instance du dataset"""
        if name not in cls._datasets:
            raise ValueError(f"Dataset non trouvé: {name}. Disponibles: {list(cls._datasets.keys())}")
        
        return cls._datasets[name](data_dir=data_dir, **kwargs)
    
    @classmethod
    def list_datasets(cls):
        """Liste tous les datasets disponibles"""
        return list(cls._datasets.keys())
    
    @classmethod
    def get_metadata(cls, name: str):
        """Retourne les métadonnées d'un dataset"""
        if name not in cls._datasets:
            raise ValueError(f"Dataset non trouvé: {name}")
        
        # Créer une instance temporaire pour obtenir les métadonnées
        temp_instance = cls._datasets[name](data_dir="/tmp")
        return temp_instance.get_metadata()
