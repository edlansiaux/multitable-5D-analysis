"""
Loader pour les datasets MIMIC-III et MIMIC-IV
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .base import MT5DDataset, DatasetMetadata, DatasetType

class MIMICLoader(MT5DDataset):
    """Chargeur pour MIMIC-III/IV (Medical Information Mart for Intensive Care)"""
    
    def __init__(self, data_dir: str, version: str = "iv", download: bool = False):
        super().__init__(data_dir, download)
        self.version = version  # "iii" ou "iv"
        self.is_loaded = False
    
    def download(self):
        """Télécharge MIMIC-IV depuis PhysioNet"""
        # Note: MIMIC nécessite une certification PhysioNet
        # Cette fonction est un placeholder
        print("MIMIC nécessite une certification PhysioNet.")
        print("Veuillez télécharger manuellement depuis:")
        print("https://physionet.org/content/mimiciv/")
        print("\nAprès téléchargement, extrayez dans:", self.data_dir)
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """Charge les tables MIMIC"""
        
        if self.is_loaded:
            return self.tables
        
        mimic_dir = self.data_dir / f"mimic-{self.version}"
        
        if not mimic_dir.exists():
            raise FileNotFoundError(
                f"Répertoire MIMIC non trouvé: {mimic_dir}. "
                f"Veuillez télécharger MIMIC-{self.version.upper()}."
            )
        
        # Charger les tables principales selon la version
        if self.version == "iv":
            self._load_mimic_iv(mimic_dir)
        elif self.version == "iii":
            self._load_mimic_iii(mimic_dir)
        else:
            raise ValueError(f"Version non supportée: {self.version}")
        
        self.is_loaded = True
        return self.tables
    
    def _load_mimic_iv(self, mimic_dir: Path):
        """Charge MIMIC-IV"""
        
        # Core module
        core_dir = mimic_dir / "core"
        if core_dir.exists():
            print("Chargement module core...")
            
            # Patients
            patients_path = core_dir / "patients.csv.gz"
            if patients_path.exists():
                self.tables['patients'] = pd.read_csv(patients_path, compression='gzip')
                print(f"  - Patients: {len(self.tables['patients'])} lignes")
            
            # Admissions
            admissions_path = core_dir / "admissions.csv.gz"
            if admissions_path.exists():
                self.tables['admissions'] = pd.read_csv(admissions_path, compression='gzip')
                print(f"  - Admissions: {len(self.tables['admissions'])} lignes")
            
            # Transfers
            transfers_path = core_dir / "transfers.csv.gz"
            if transfers_path.exists():
                self.tables['transfers'] = pd.read_csv(transfers_path, compression='gzip')
                print(f"  - Transfers: {len(self.tables['transfers'])} lignes")
        
        # ICU module
        icu_dir = mimic_dir / "icu"
        if icu_dir.exists():
            print("Chargement module ICU...")
            
            # Chartevents (échantillon pour performance)
            chartevents_path = icu_dir / "chartevents.csv.gz"
            if chartevents_path.exists():
                # Charger seulement un échantillon pour la démo
                try:
                    # Lire les 10000 premières lignes
                    self.tables['chartevents'] = pd.read_csv(
                        chartevents_path, 
                        compression='gzip',
                        nrows=10000
                    )
                    print(f"  - Chartevents (échantillon): {len(self.tables['chartevents'])} lignes")
                except Exception as e:
                    print(f"  - Erreur chargement chartevents: {e}")
            
            # D_Items
            d_items_path = icu_dir / "d_items.csv.gz"
            if d_items_path.exists():
                self.tables['d_items'] = pd.read_csv(d_items_path, compression='gzip')
                print(f"  - D_Items: {len(self.tables['d_items'])} lignes")
        
        # Hosp module
        hosp_dir = mimic_dir / "hosp"
        if hosp_dir.exists():
            print("Chargement module hosp...")
            
            # Labevents
            labevents_path = hosp_dir / "labevents.csv.gz"
            if labevents_path.exists():
                try:
                    self.tables['labevents'] = pd.read_csv(
                        labevents_path,
                        compression='gzip',
                        nrows=50000
                    )
                    print(f"  - Labevents (échantillon): {len(self.tables['labevents'])} lignes")
                except Exception as e:
                    print(f"  - Erreur chargement labevents: {e}")
            
            # D_Labitems
            d_labitems_path = hosp_dir / "d_labitems.csv.gz"
            if d_labitems_path.exists():
                self.tables['d_labitems'] = pd.read_csv(d_labitems_path, compression='gzip')
                print(f"  - D_Labitems: {len(self.tables['d_labitems'])} lignes")
            
            # Diagnoses (ICD codes)
            diagnoses_path = hosp_dir / "diagnoses_icd.csv.gz"
            if diagnoses_path.exists():
                try:
                    self.tables['diagnoses'] = pd.read_csv(
                        diagnoses_path,
                        compression='gzip',
                        nrows=20000
                    )
                    print(f"  - Diagnoses (échantillon): {len(self.tables['diagnoses'])} lignes")
                except Exception as e:
                    print(f"  - Erreur chargement diagnoses: {e}")
            
            # D_ICD_Diagnoses
            d_icd_path = hosp_dir / "d_icd_diagnoses.csv.gz"
            if d_icd_path.exists():
                self.tables['d_icd_diagnoses'] = pd.read_csv(d_icd_path, compression='gzip')
                print(f"  - D_ICD_Diagnoses: {len(self.tables['d_icd_diagnoses'])} lignes")
    
    def _load_mimic_iii(self, mimic_dir: Path):
        """Charge MIMIC-III (version simplifiée)"""
        
        print("Chargement MIMIC-III...")
        
        # Patients
        patients_path = mimic_dir / "PATIENTS.csv.gz"
        if patients_path.exists():
            self.tables['patients'] = pd.read_csv(patients_path, compression='gzip')
            print(f"  - Patients: {len(self.tables['patients'])} lignes")
        
        # Admissions
        admissions_path = mimic_dir / "ADMISSIONS.csv.gz"
        if admissions_path.exists():
            self.tables['admissions'] = pd.read_csv(admissions_path, compression='gzip')
            print(f"  - Admissions: {len(self.tables['admissions'])} lignes")
        
        # Chartevents (échantillon)
        chartevents_path = mimic_dir / "CHARTEVENTS.csv.gz"
        if chartevents_path.exists():
            try:
                self.tables['chartevents'] = pd.read_csv(
                    chartevents_path,
                    compression='gzip',
                    nrows=10000
                )
                print(f"  - Chartevents (échantillon): {len(self.tables['chartevents'])} lignes")
            except Exception as e:
                print(f"  - Erreur chargement chartevents: {e}")
        
        # Labevents (échantillon)
        labevents_path = mimic_dir / "LABEVENTS.csv.gz"
        if labevents_path.exists():
            try:
                self.tables['labevents'] = pd.read_csv(
                    labevents_path,
                    compression='gzip',
                    nrows=50000
                )
                print(f"  - Labevents (échantillon): {len(self.tables['labevents'])} lignes")
            except Exception as e:
                print(f"  - Erreur chargement labevents: {e}")
        
        # Diagnoses_ICD
        diagnoses_path = mimic_dir / "DIAGNOSES_ICD.csv.gz"
        if diagnoses_path.exists():
            try:
                self.tables['diagnoses'] = pd.read_csv(
                    diagnoses_path,
                    compression='gzip',
                    nrows=20000
                )
                print(f"  - Diagnoses (échantillon): {len(self.tables['diagnoses'])} lignes")
            except Exception as e:
                print(f"  - Erreur chargement diagnoses: {e}")
    
    def get_relationships(self) -> List[Tuple]:
        """Retourne les relations entre tables MIMIC"""
        
        relationships = []
        
        if self.version == "iv":
            # Relations MIMIC-IV
            if 'patients' in self.tables and 'admissions' in self.tables:
                relationships.append(
                    ('admissions', 'subject_id', 'patients', 'subject_id', 'patient_admission')
                )
            
            if 'admissions' in self.tables and 'transfers' in self.tables:
                relationships.append(
                    ('transfers', 'hadm_id', 'admissions', 'hadm_id', 'admission_transfer')
                )
            
            if 'chartevents' in self.tables and 'd_items' in self.tables:
                relationships.append(
                    ('chartevents', 'itemid', 'd_items', 'itemid', 'chart_item')
                )
            
            if 'labevents' in self.tables and 'd_labitems' in self.tables:
                relationships.append(
                    ('labevents', 'itemid', 'd_labitems', 'itemid', 'lab_item')
                )
            
            if 'diagnoses' in self.tables and 'd_icd_diagnoses' in self.tables:
                relationships.append(
                    ('diagnoses', 'icd_code', 'd_icd_diagnoses', 'icd_code', 'diagnosis_code')
                )
            
            if 'diagnoses' in self.tables and 'admissions' in self.tables:
                relationships.append(
                    ('diagnoses', 'hadm_id', 'admissions', 'hadm_id', 'admission_diagnosis')
                )
        
        elif self.version == "iii":
            # Relations MIMIC-III
            if 'patients' in self.tables and 'admissions' in self.tables:
                relationships.append(
                    ('admissions', 'SUBJECT_ID', 'patients', 'SUBJECT_ID', 'patient_admission')
                )
            
            if 'chartevents' in self.tables and 'admissions' in self.tables:
                relationships.append(
                    ('chartevents', 'HADM_ID', 'admissions', 'HADM_ID', 'admission_chart')
                )
            
            if 'labevents' in self.tables and 'admissions' in self.tables:
                relationships.append(
                    ('labevents', 'HADM_ID', 'admissions', 'HADM_ID', 'admission_lab')
                )
            
            if 'diagnoses' in self.tables and 'admissions' in self.tables:
                relationships.append(
                    ('diagnoses', 'HADM_ID', 'admissions', 'HADM_ID', 'admission_diagnosis')
                )
        
        return relationships
    
    def get_metadata(self) -> DatasetMetadata:
        """Retourne les métadonnées MIMIC"""
        
        if not self.metadata:
            self.metadata = DatasetMetadata(
                name=f"MIMIC-{self.version.upper()}",
                type=DatasetType.MEDICAL,
                description=f"Medical Information Mart for Intensive Care {self.version.upper()} - Database de soins intensifs",
                source="PhysioNet",
                license="PhysioNet Credentialed Health Data License",
                num_tables=len(self.tables),
                total_rows=sum(len(df) for df in self.tables.values()),
                total_columns=sum(len(df.columns) for df in self.tables.values()),
                has_temporal_data=True,
                has_relationships=True,
                download_url=f"https://physionet.org/content/mimic{self.version}/",
                citation="Johnson, A., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023).",
                version="2.2" if self.version == "iv" else "1.4"
            )
        
        return self.metadata
    
    def create_clinical_task(self, task: str = "mortality_prediction"):
        """Crée une tâche clinique spécifique"""
        
        if not self.is_loaded:
            self.load()
        
        if task == "mortality_prediction":
            return self._create_mortality_prediction_task()
        elif task == "readmission_prediction":
            return self._create_readmission_prediction_task()
        elif task == "length_of_stay_prediction":
            return self._create_los_prediction_task()
        else:
            raise ValueError(f"Tâche non supportée: {task}")
    
    def _create_mortality_prediction_task(self) -> pd.DataFrame:
        """Crée une tâche de prédiction de mortalité"""
        
        if 'patients' not in self.tables or 'admissions' not in self.tables:
            raise ValueError("Tables patients et admissions nécessaires")
        
        # Fusionner patients et admissions
        df = pd.merge(
            self.tables['patients'][['subject_id', 'gender', 'anchor_age']],
            self.tables['admissions'][['subject_id', 'hadm_id', 'admittime', 'dischtime', 
                                      'hospital_expire_flag']],
            on='subject_id',
            how='inner'
        )
        
        # Créer la variable cible
        df['mortality'] = df['hospital_expire_flag'].astype(int)
        
        # Calculer l'âge à l'admission
        df['age'] = df['anchor_age']
        
        # Nettoyer
        df = df[['subject_id', 'hadm_id', 'gender', 'age', 'mortality']].dropna()
        
        return df
    
    def _create_readmission_prediction_task(self) -> pd.DataFrame:
        """Crée une tâche de prédiction de réadmission"""
        
        if 'admissions' not in self.tables:
            raise ValueError("Table admissions nécessaire")
        
        admissions = self.tables['admissions'].copy()
        
        # Trier par patient et date d'admission
        admissions['admittime'] = pd.to_datetime(admissions['admittime'])
        admissions = admissions.sort_values(['subject_id', 'admittime'])
        
        # Calculer le temps jusqu'à la prochaine admission
        admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)
        admissions['days_to_readmission'] = (
            admissions['next_admittime'] - admissions['admittime']
        ).dt.days
        
        # Définir la cible: réadmission dans les 30 jours
        admissions['readmission_30days'] = (
            admissions['days_to_readmission'] <= 30
        ).astype(int)
        
        # Garder seulement la dernière admission pour chaque patient (pour éviter le leakage)
        last_admissions = admissions.groupby('subject_id').last().reset_index()
        
        return last_admissions[['subject_id', 'hadm_id', 'admittime', 'readmission_30days']]
    
    def _create_los_prediction_task(self) -> pd.DataFrame:
        """Crée une tâche de prédiction de durée de séjour"""
        
        if 'admissions' not in self.tables:
            raise ValueError("Table admissions nécessaire")
        
        admissions = self.tables['admissions'].copy()
        
        # Convertir les dates
        admissions['admittime'] = pd.to_datetime(admissions['admittime'])
        admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
        
        # Calculer la durée de séjour
        admissions['length_of_stay_days'] = (
            admissions['dischtime'] - admissions['admittime']
        ).dt.days
        
        # Nettoyer (enlever les valeurs négatives ou trop grandes)
        admissions = admissions[
            (admissions['length_of_stay_days'] >= 0) & 
            (admissions['length_of_stay_days'] <= 365)
        ]
        
        return admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'length_of_stay_days']]

# Enregistrer dans le registre
from .base import DatasetRegistry
DatasetRegistry.register("mimic-iv", MIMICLoader)
DatasetRegistry.register("mimic-iii", lambda data_dir, **kwargs: MIMICLoader(data_dir, version="iii", **kwargs))
