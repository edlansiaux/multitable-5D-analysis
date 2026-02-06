import pandas as pd
from typing import Dict, Tuple, List
from .base import BaseDataset

class MIMICDataset(BaseDataset):
    """
    Chargeur pour MIMIC-IV (Medical Information Mart for Intensive Care).
    Définit les relations natives entre Patients, Admissions, Diagnostiques, etc.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.tables = {}
        self.relationships = []
        
    def load(self) -> Tuple[Dict[str, pd.DataFrame], List[Tuple]]:
        """
        Charge les tables et définit les relations explicites.
        """
        # Liste des tables critiques pour l'analyse 5D
        required_tables = [
            'patients', 'admissions', 'diagnoses_icd', 
            'labevents', 'prescriptions', 'transfers'
        ]
        
        # Chargement (simulation d'appel à IO)
        # En production, utiliser DataLoader.load_schema_folder(self.root_dir)
        print(f"Chargement de MIMIC-IV depuis {self.root_dir}...")
        
        # Définition du schéma relationnel (Table A, Col A, Table B, Col B, Type)
        # Basé sur la documentation officielle MIMIC-IV
        self.relationships = [
            # Patients <-> Admissions
            ('patients', 'subject_id', 'admissions', 'subject_id', 'one_to_many'),
            
            # Admissions <-> Diagnostiques (Haute cardinalité)
            ('admissions', 'hadm_id', 'diagnoses_icd', 'hadm_id', 'one_to_many'),
            
            # Admissions <-> Lab Events (Séries temporelles)
            ('admissions', 'hadm_id', 'labevents', 'hadm_id', 'one_to_many'),
            
            # Admissions <-> Prescriptions
            ('admissions', 'hadm_id', 'prescriptions', 'hadm_id', 'one_to_many'),
            
            # Admissions <-> Transferts (Flux patient)
            ('admissions', 'hadm_id', 'transfers', 'hadm_id', 'one_to_many')
        ]
        
        # Note: Le chargement réel des CSVs est laissé à l'utilisateur via le DataLoader
        # pour éviter de dépendre de fichiers locaux non présents.
        return self.tables, self.relationships

    def get_temporal_columns(self) -> Dict[str, str]:
        """Retourne les colonnes temporelles clés pour chaque table."""
        return {
            'admissions': 'admittime',
            'labevents': 'charttime',
            'prescriptions': 'starttime',
            'transfers': 'intime'
        }
