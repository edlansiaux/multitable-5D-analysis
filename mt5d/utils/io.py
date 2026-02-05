"""
Module d'IO pour lire/écrire des données multitable
"""
import pandas as pd
import numpy as np
import json
import yaml
import pickle
import dill
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings

class DataLoader:
    """Chargeur de données multitable"""
    
    SUPPORTED_FORMATS = {
        'csv': 'Comma Separated Values',
        'parquet': 'Apache Parquet',
        'feather': 'Feather Format',
        'json': 'JSON',
        'hdf5': 'HDF5 Format',
        'sql': 'SQL Database'
    }
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else None
        
    def load_from_directory(self, directory: str, 
                           format: str = 'auto',
                           recursive: bool = False) -> Dict[str, pd.DataFrame]:
        """Charge toutes les tables d'un répertoire"""
        
        data_dir = Path(directory)
        tables = {}
        
        # Détecter le format si auto
        if format == 'auto':
            format = self._detect_format(data_dir)
        
        # Pattern de recherche
        if recursive:
            pattern = f"**/*.{format}"
        else:
            pattern = f"*.{format}"
        
        # Charger chaque fichier
        for file_path in data_dir.glob(pattern):
            table_name = file_path.stem  # Nom sans extension
            
            try:
                df = self._load_single_file(file_path, format)
                tables[table_name] = df
                print(f"✓ Chargé: {table_name} ({len(df)} lignes, {len(df.columns)} colonnes)")
            except Exception as e:
                print(f"✗ Erreur chargement {file_path}: {e}")
        
        return tables
    
    def load_relationships(self, filepath: str) -> List[tuple]:
        """Charge les relations depuis un fichier"""
        
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif filepath.suffix == '.csv':
            data = pd.read_csv(filepath).to_dict('records')
        else:
            raise ValueError(f"Format non supporté: {filepath.suffix}")
        
        # Convertir en format standard
        relationships = []
        for rel in data:
            if isinstance(rel, dict):
                relationships.append((
                    rel.get('source_table'),
                    rel.get('source_column'),
                    rel.get('target_table'),
                    rel.get('target_column'),
                    rel.get('relation_type', 'related_to')
                ))
            elif isinstance(rel, (list, tuple)) and len(rel) >= 4:
                rel_type = rel[4] if len(rel) > 4 else 'related_to'
                relationships.append((rel[0], rel[1], rel[2], rel[3], rel_type))
        
        return relationships
    
    def save_results(self, results: Dict[str, Any], output_dir: str, 
                    format: str = 'pickle'):
        """Sauvegarde les résultats d'analyse"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder chaque composant
        for key, value in results.items():
            if key == 'hypergraph':
                self._save_hypergraph(value, output_dir / f"{key}.pkl")
            elif key == 'model' and hasattr(value, 'state_dict'):
                self._save_model(value, output_dir / f"{key}.pt")
            elif isinstance(value, pd.DataFrame):
                value.to_parquet(output_dir / f"{key}.parquet")
            elif isinstance(value, (dict, list)):
                with open(output_dir / f"{key}.json", 'w') as f:
                    json.dump(value, f, indent=2, default=self._json_serializer)
            else:
                # Sauvegarde générique avec pickle
                with open(output_dir / f"{key}.pkl", 'wb') as f:
                    pickle.dump(value, f)
        
        print(f"Résultats sauvegardés dans: {output_dir}")
    
    def load_results(self, input_dir: str) -> Dict[str, Any]:
        """Charge les résultats sauvegardés"""
        
        input_dir = Path(input_dir)
        results = {}
        
        for file_path in input_dir.glob("*"):
            key = file_path.stem
            
            try:
                if file_path.suffix == '.pkl':
                    with open(file_path, 'rb') as f:
                        results[key] = pickle.load(f)
                elif file_path.suffix == '.pt':
                    results[key] = self._load_model(file_path)
                elif file_path.suffix == '.parquet':
                    results[key] = pd.read_parquet(file_path)
                elif file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        results[key] = json.load(f)
                else:
                    print(f"Format non reconnu: {file_path}")
            except Exception as e:
                print(f"Erreur chargement {file_path}: {e}")
        
        return results
    
    def export_to_sql(self, tables: Dict[str, pd.DataFrame], 
                     connection_string: str,
                     schema: str = 'mt5d'):
        """Exporte les tables vers une base SQL"""
        
        import sqlalchemy
        
        engine = sqlalchemy.create_engine(connection_string)
        
        with engine.connect() as conn:
            # Créer le schéma
            conn.execute(sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            
            # Exporter chaque table
            for table_name, df in tables.items():
                df.to_sql(
                    name=table_name,
                    con=engine,
                    schema=schema,
                    if_exists='replace',
                    index=False
                )
                print(f"✓ Exporté: {schema}.{table_name}")
    
    def _load_single_file(self, filepath: Path, format: str) -> pd.DataFrame:
        """Charge un seul fichier"""
        
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'parquet':
            return pd.read_parquet(filepath)
        elif format == 'feather':
            return pd.read_feather(filepath)
        elif format == 'json':
            return pd.read_json(filepath)
        elif format == 'hdf5':
            return pd.read_hdf(filepath)
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    def _detect_format(self, directory: Path) -> str:
        """Détecte le format dominant dans le répertoire"""
        
        extensions = {}
        for file_path in directory.glob("*.*"):
            ext = file_path.suffix[1:]  # Enlever le point
            if ext in self.SUPPORTED_FORMATS:
                extensions[ext] = extensions.get(ext, 0) + 1
        
        if not extensions:
            raise FileNotFoundError(f"Aucun fichier supporté dans {directory}")
        
        return max(extensions.items(), key=lambda x: x[1])[0]
    
    def _save_hypergraph(self, hypergraph, filepath: Path):
        """Sauvegarde un hypergraphe DGL"""
        import dgl
        dgl.save_graphs(str(filepath), [hypergraph])
    
    def _load_model(self, filepath: Path):
        """Charge un modèle PyTorch"""
        import torch
        # Note: L'architecture du modèle doit être définie
        return torch.load(filepath)
    
    def _json_serializer(self, obj):
        """Sérialiseur JSON pour objets personnalisés"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        else:
            raise TypeError(f"Type non sérialisable: {type(obj)}")

class DataValidator:
    """Validateur de données multitable"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate(self, tables: Dict[str, pd.DataFrame], 
                relationships: List[tuple] = None) -> bool:
        """Valide la cohérence des données"""
        
        self.errors.clear()
        self.warnings.clear()
        
        # 1. Validation des tables
        self._validate_tables(tables)
        
        # 2. Validation des relations
        if relationships:
            self._validate_relationships(tables, relationships)
        
        # 3. Validation des données
        self._validate_data_quality(tables)
        
        # Afficher les résultats
        if self.errors:
            print("✗ ERREURS DE VALIDATION:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("⚠ AVERTISSEMENTS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        return len(self.errors) == 0
    
    def _validate_tables(self, tables: Dict[str, pd.DataFrame]):
        """Valide la structure des tables"""
        
        if not tables:
            self.errors.append("Aucune table fournie")
            return
        
        for table_name, df in tables.items():
            # Vérifier que c'est un DataFrame
            if not isinstance(df, pd.DataFrame):
                self.errors.append(f"{table_name}: n'est pas un DataFrame")
                continue
            
            # Vérifier les doublons de noms de colonnes
            duplicates = df.columns[df.columns.duplicated()].tolist()
            if duplicates:
                self.errors.append(f"{table_name}: colonnes en double: {duplicates}")
            
            # Vérifier les valeurs manquantes
            missing_counts = df.isnull().sum()
            high_missing = missing_counts[missing_counts > len(df) * 0.5]  # > 50%
            if not high_missing.empty:
                self.warnings.append(
                    f"{table_name}: colonnes avec >50% valeurs manquantes: "
                    f"{list(high_missing.index)}"
                )
            
            # Vérifier les types de données
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 20:
                self.warnings.append(
                    f"{table_name}: nombreuses colonnes textuelles ({len(object_cols)})"
                )
    
    def _validate_relationships(self, tables: Dict[str, pd.DataFrame], 
                              relationships: List[tuple]):
        """Valide la cohérence des relations"""
        
        for rel in relationships:
            if len(rel) < 4:
                self.errors.append(f"Relation invalide: {rel}")
                continue
            
            src_table, src_col, tgt_table, tgt_col = rel[:4]
            
            # Vérifier que les tables existent
            if src_table not in tables:
                self.errors.append(f"Table source non trouvée: {src_table}")
                continue
            
            if tgt_table not in tables:
                self.errors.append(f"Table cible non trouvée: {tgt_table}")
                continue
            
            # Vérifier que les colonnes existent
            if src_col not in tables[src_table].columns:
                self.errors.append(
                    f"Colonne {src_col} non trouvée dans {src_table}"
                )
            
            if tgt_col not in tables[tgt_table].columns:
                self.errors.append(
                    f"Colonne {tgt_col} non trouvée dans {tgt_table}"
                )
            
            # Vérifier la compatibilité des types
            if src_col in tables[src_table].columns and tgt_col in tables[tgt_table].columns:
                src_dtype = tables[src_table][src_col].dtype
                tgt_dtype = tables[tgt_table][tgt_col].dtype
                
                if not self._are_dtypes_compatible(src_dtype, tgt_dtype):
                    self.warnings.append(
                        f"Types incompatibles: {src_table}.{src_col} ({src_dtype}) "
                        f"-> {tgt_table}.{tgt_col} ({tgt_dtype})"
                    )
    
    def _validate_data_quality(self, tables: Dict[str, pd.DataFrame]):
        """Valide la qualité des données"""
        
        for table_name, df in tables.items():
            # Vérifier les outliers dans les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Utiliser IQR pour détecter les outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                
                if len(outliers) > len(df) * 0.1:  # > 10% outliers
                    self.warnings.append(
                        f"{table_name}.{col}: {len(outliers)} outliers détectés "
                        f"({len(outliers)/len(df)*100:.1f}%)"
                    )
            
            # Vérifier les valeurs uniques pour les colonnes catégorielles
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count == 1:
                    self.warnings.append(
                        f"{table_name}.{col}: seule une valeur unique"
                    )
                elif unique_count == len(df):
                    self.warnings.append(
                        f"{table_name}.{col}: toutes les valeurs sont uniques "
                        f"(clé potentielle)"
                    )
    
    def _are_dtypes_compatible(self, dtype1, dtype2) -> bool:
        """Vérifie si deux types de données sont compatibles"""
        
        # Groupes compatibles
        numeric_types = [np.dtype('int64'), np.dtype('float64'), 
                        np.dtype('int32'), np.dtype('float32')]
        
        if dtype1 in numeric_types and dtype2 in numeric_types:
            return True
        
        # Types objets/catégoriels sont compatibles entre eux
        object_types = [np.dtype('object'), np.dtype('category')]
        if dtype1 in object_types and dtype2 in object_types:
            return True
        
        # Même type
        return dtype1 == dtype2

def create_sample_dataset(output_dir: str, n_samples: int = 1000):
    """Crée un dataset d'exemple"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Patients
    patients = pd.DataFrame({
        'patient_id': range(n_samples),
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['M', 'F', 'O'], n_samples, p=[0.49, 0.49, 0.02]),
        'height_cm': np.random.normal(170, 10, n_samples).astype(int),
        'weight_kg': np.random.normal(70, 15, n_samples).astype(int),
        'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], 
                                      n_samples),
        'registration_date': pd.to_datetime(
            np.random.choice(pd.date_range('2010-01-01', '2023-01-01', freq='D'), 
                           n_samples)
        )
    })
    
    # Diagnostiques
    n_diagnoses = n_samples * 2
    diagnoses = pd.DataFrame({
        'diagnosis_id': range(n_diagnoses),
        'patient_id': np.random.choice(patients['patient_id'], n_diagnoses),
        'code': np.random.choice([f'I{i}' for i in range(10, 100, 10)], n_diagnoses),
        'description': np.random.choice([
            'Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 
            'Migraine', 'Depression', 'Anxiety', 'Obesity'
        ], n_diagnoses),
        'diagnosis_date': pd.to_datetime(
            np.random.choice(pd.date_range('2020-01-01', '2023-12-31', freq='D'), 
                           n_diagnoses)
        ),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_diagnoses, 
                                    p=[0.6, 0.3, 0.1])
    })
    
    # Visites
    n_visits = n_samples * 3
    visits = pd.DataFrame({
        'visit_id': range(n_visits),
        'patient_id': np.random.choice(patients['patient_id'], n_visits),
        'visit_date': pd.to_datetime(
            np.random.choice(pd.date_range('2021-01-01', '2023-12-31', freq='H'), 
                           n_visits)
        ),
        'systolic_bp': np.random.normal(120, 20, n_visits).astype(int),
        'diastolic_bp': np.random.normal(80, 15, n_visits).astype(int),
        'heart_rate': np.random.normal(75, 15, n_visits).astype(int),
        'temperature': np.random.normal(36.6, 0.5, n_visits),
        'doctor_id': np.random.randint(1, 50, n_visits)
    })
    
    # Médicaments
    n_prescriptions = n_samples * 4
    medications = pd.DataFrame({
        'prescription_id': range(n_prescriptions),
        'patient_id': np.random.choice(patients['patient_id'], n_prescriptions),
        'medication_name': np.random.choice([
            'Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin',
            'Levothyroxine', 'Metoprolol', 'Amlodipine', 'Omeprazole'
        ], n_prescriptions),
        'dosage_mg': np.random.choice([5, 10, 20, 50, 100, 200, 500], n_prescriptions),
        'frequency': np.random.choice(['QD', 'BID', 'TID', 'QID'], n_prescriptions),
        'start_date': pd.to_datetime(
            np.random.choice(pd.date_range('2022-01-01', '2023-12-31', freq='D'), 
                           n_prescriptions)
        ),
        'end_date': pd.to_datetime(
            np.random.choice(pd.date_range('2023-01-01', '2024-12-31', freq='D'), 
                           n_prescriptions)
        )
    })
    
    # Sauvegarder
    patients.to_parquet(output_dir / 'patients.parquet')
    diagnoses.to_parquet(output_dir / 'diagnoses.parquet')
    visits.to_parquet(output_dir / 'visits.parquet')
    medications.to_parquet(output_dir / 'medications.parquet')
    
    # Créer les relations
    relationships = [
        ('diagnoses', 'patient_id', 'patients', 'patient_id', 'patient_diagnosis'),
        ('visits', 'patient_id', 'patients', 'patient_id', 'patient_visit'),
        ('medications', 'patient_id', 'patients', 'patient_id', 'patient_medication')
    ]
    
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
    
    print(f"✓ Dataset créé dans: {output_dir}")
    print(f"  - Patients: {len(patients)}")
    print(f"  - Diagnostiques: {len(diagnoses)}")
    print(f"  - Visites: {len(visits)}")
    print(f"  - Médicaments: {len(medications)}")
