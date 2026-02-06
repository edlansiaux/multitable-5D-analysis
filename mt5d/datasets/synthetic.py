import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

class SyntheticMultiTableGenerator:
    """
    Générateur de données synthétiques complexes pour tester le framework 5D.
    Simule : Volume, Multi-variables, Haute Cardinalité, Relations, Temporalité.
    """
    
    def __init__(self, num_patients=1000):
        self.num_patients = num_patients
        self.rng = np.random.default_rng(42)
        
    def generate(self) -> Tuple[Dict[str, pd.DataFrame], List[Tuple]]:
        print("Génération de données synthétiques multi-tables (Mock MIMIC)...")
        
        # Table 1: Patients (Core entity)
        patients = pd.DataFrame({
            'subject_id': range(self.num_patients),
            'age': self.rng.integers(18, 90, self.num_patients),
            'gender': self.rng.choice(['M', 'F'], self.num_patients),
            'static_feature_1': self.rng.normal(0, 1, self.num_patients)
        })
        
        # Table 2: Admissions (One-to-Many avec Patients, Temporel)
        num_admissions = self.num_patients * 2
        admissions = pd.DataFrame({
            'hadm_id': range(num_admissions),
            'subject_id': self.rng.choice(patients['subject_id'], num_admissions),
            'admittime': pd.date_range(start='2020-01-01', periods=num_admissions, freq='H'),
            'admission_type': self.rng.choice(['EMERGENCY', 'ELECTIVE'], num_admissions)
        })
        
        # Table 3: Diagnoses (Haute Cardinalité - ICD Codes) [cite: 39]
        num_diag = num_admissions * 5
        icd_codes = [f"ICD_{i}" for i in range(2000)] # 2000 codes uniques
        diagnoses = pd.DataFrame({
            'hadm_id': self.rng.choice(admissions['hadm_id'], num_diag),
            'icd_code': self.rng.choice(icd_codes, num_diag), # High cardinality
            'seq_num': self.rng.integers(1, 10, num_diag)
        })
        
        # Table 4: Lab Events (Mesures répétées irrégulières - Dim 5) [cite: 42]
        num_labs = num_admissions * 20
        labs = pd.DataFrame({
            'hadm_id': self.rng.choice(admissions['hadm_id'], num_labs),
            'charttime': pd.date_range(start='2020-01-01', periods=num_labs, freq='min'),
            'itemid': self.rng.integers(50000, 50100, num_labs),
            'valuenum': self.rng.normal(37, 2, num_labs) # Ex: Température
        })
        
        tables = {
            'patients': patients,
            'admissions': admissions,
            'diagnoses': diagnoses,
            'labevents': labs
        }
        
        # Définition des relations (Schema)
        relationships = [
            ('patients', 'subject_id', 'admissions', 'subject_id', 'one_to_many'),
            ('admissions', 'hadm_id', 'diagnoses', 'hadm_id', 'one_to_many'),
            ('admissions', 'hadm_id', 'labevents', 'hadm_id', 'one_to_many')
        ]
        
        return tables, relationships
