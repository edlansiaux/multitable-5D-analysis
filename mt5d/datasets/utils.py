"""
Utilitaires pour télécharger et prétraiter les datasets
"""
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def download_dataset(url: str, output_dir: str, extract: bool = True):
    """
    Télécharge un dataset depuis une URL
    
    Args:
        url: URL du dataset
        output_dir: Répertoire de sortie
        extract: Si True, extrait les archives
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extraire le nom du fichier depuis l'URL
    filename = url.split('/')[-1]
    filepath = output_dir / filename
    
    print(f"Téléchargement depuis: {url}")
    print(f"Vers: {filepath}")
    
    # Téléchargement avec barre de progression
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
        
        print(f"✓ Téléchargement terminé: {filepath}")
        
        # Extraction si demandée
        if extract:
            extract_archive(filepath, output_dir)
    
    except Exception as e:
        print(f"✗ Erreur téléchargement: {e}")
        raise

def extract_archive(filepath: Path, output_dir: Path):
    """
    Extrait une archive (zip, tar, gz)
    
    Args:
        filepath: Chemin de l'archive
        output_dir: Répertoire d'extraction
    """
    
    print(f"Extraction: {filepath}")
    
    try:
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                print(f"✓ Archive ZIP extraite")
        
        elif filepath.suffix in ['.tar', '.gz', '.bz2', '.xz']:
            if '.tar' in filepath.suffixes:
                # Fichier tar (peut être compressé)
                mode = 'r'
                if '.gz' in filepath.suffixes:
                    mode = 'r:gz'
                elif '.bz2' in filepath.suffixes:
                    mode = 'r:bz2'
                elif '.xz' in filepath.suffixes:
                    mode = 'r:xz'
                
                with tarfile.open(filepath, mode) as tar_ref:
                    tar_ref.extractall(output_dir)
                    print(f"✓ Archive TAR extraite")
            
            elif filepath.suffix == '.gz' and '.tar' not in filepath.suffixes:
                # Fichier gzip simple
                output_file = output_dir / filepath.stem
                with gzip.open(filepath, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"✓ Fichier GZ décompressé: {output_file}")
        
        else:
            print(f"✗ Format d'archive non supporté: {filepath.suffix}")
    
    except Exception as e:
        print(f"✗ Erreur extraction: {e}")

def preprocess_dataset(tables: Dict[str, pd.DataFrame], 
                      preprocessing_steps: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Prétraite les tables selon des étapes spécifiques
    
    Args:
        tables: Dictionnaire de DataFrames
        preprocessing_steps: Étapes de prétraitement par table
            Ex: {'patients': ['fill_missing', 'encode_categorical']}
    
    Returns:
        Tables prétraitées
    """
    
    processed_tables = {}
    
    for table_name, df in tables.items():
        print(f"Prétraitement table: {table_name}")
        
        if table_name not in preprocessing_steps:
            # Pas de prétraitement spécifié, appliquer les étapes par défaut
            df_processed = _apply_default_preprocessing(df)
        else:
            # Appliquer les étapes spécifiées
            df_processed = df.copy()
            steps = preprocessing_steps[table_name]
            
            for step in steps:
                if step == 'fill_missing':
                    df_processed = _fill_missing_values(df_processed)
                elif step == 'encode_categorical':
                    df_processed = _encode_categorical(df_processed)
                elif step == 'normalize_numeric':
                    df_processed = _normalize_numeric(df_processed)
                elif step == 'extract_temporal_features':
                    df_processed = _extract_temporal_features(df_processed)
                elif step == 'remove_outliers':
                    df_processed = _remove_outliers(df_processed)
                elif step == 'feature_engineering':
                    df_processed = _feature_engineering(df_processed)
                else:
                    print(f"  ⚠ Étape non reconnue: {step}")
        
        processed_tables[table_name] = df_processed
        print(f"  ✓ {len(df)} -> {len(df_processed)} lignes, {len(df_processed.columns)} colonnes")
    
    return processed_tables

def _apply_default_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le prétraitement par défaut"""
    
    df_processed = df.copy()
    
    # 1. Remplir les valeurs manquantes
    df_processed = _fill_missing_values(df_processed)
    
    # 2. Encoder les variables catégorielles
    df_processed = _encode_categorical(df_processed)
    
    # 3. Extraire les features temporelles si présentes
    df_processed = _extract_temporal_features(df_processed)
    
    return df_processed

def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remplit les valeurs manquantes"""
    
    df_filled = df.copy()
    
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                # Pour les colonnes numériques: médiane
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            elif pd.api.types.is_categorical_dtype(df_filled[col]) or \
                 pd.api.types.is_object_dtype(df_filled[col]):
                # Pour les colonnes catégorielles: mode
                mode_val = df_filled[col].mode()
                if not mode_val.empty:
                    df_filled[col] = df_filled[col].fillna(mode_val.iloc[0])
                else:
                    df_filled[col] = df_filled[col].fillna('Unknown')
    
    return df_filled

def _encode_categorical(df: pd.DataFrame, max_categories: int = 50) -> pd.DataFrame:
    """Encode les variables catégorielles"""
    
    df_encoded = df.copy()
    
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        unique_count = df_encoded[col].nunique()
        
        if unique_count == 2:
            # Binaire: label encoding
            df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        elif 2 < unique_count <= max_categories:
            # One-hot encoding pour catégories avec nombre raisonnable
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
        
        else:
            # Haute cardinalité: hash encoding ou target encoding
            # Pour l'instant, on garde tel quel
            pass
    
    return df_encoded

def _normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes numériques"""
    
    df_normalized = df.copy()
    
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Standard scaling (z-score)
        mean = df_normalized[col].mean()
        std = df_normalized[col].std()
        
        if std > 0:  # Éviter la division par zéro
            df_normalized[col] = (df_normalized[col] - mean) / std
    
    return df_normalized

def _extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait des features temporelles des colonnes datetime"""
    
    df_with_features = df.copy()
    
    datetime_cols = df_with_features.select_dtypes(include=['datetime']).columns
    
    for col in datetime_cols:
        # Extraire différentes composantes temporelles
        df_with_features[f'{col}_year'] = df_with_features[col].dt.year
        df_with_features[f'{col}_month'] = df_with_features[col].dt.month
        df_with_features[f'{col}_day'] = df_with_features[col].dt.day
        df_with_features[f'{col}_hour'] = df_with_features[col].dt.hour
        df_with_features[f'{col}_dayofweek'] = df_with_features[col].dt.dayofweek
        df_with_features[f'{col}_is_weekend'] = df_with_features[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Pour les dates seulement (pas heures), extraire la saison
        if df_with_features[col].dt.hour.max() == 0:  # Supposer que c'est une date pure
            df_with_features[f'{col}_season'] = df_with_features[col].dt.month.map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            })
    
    return df_with_features

def _remove_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """Supprime les outliers"""
    
    if method == 'iqr':
        return _remove_outliers_iqr(df)
    elif method == 'zscore':
        return _remove_outliers_zscore(df)
    else:
        return df

def _remove_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """Supprime les outliers avec la méthode IQR"""
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Marquer les outliers
        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        
        # Remplacer les outliers par les bornes
        df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
        df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
    
    return df_clean

def _remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    """Supprime les outliers avec la méthode Z-score"""
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        
        # Remplacer les outliers par la médiane
        median_val = df_clean[col].median()
        df_clean.loc[z_scores > threshold, col] = median_val
    
    return df_clean

def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée de nouvelles features"""
    
    df_with_features = df.copy()
    
    # Interactions entre colonnes numériques
    numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Créer quelques interactions
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                df_with_features[f'{col1}_x_{col2}'] = df_with_features[col1] * df_with_features[col2]
                df_with_features[f'{col1}_div_{col2}'] = df_with_features[col1] / (df_with_features[col2] + 1e-8)
    
    # Polynomial features pour les colonnes importantes
    important_numeric_cols = numeric_cols[:min(3, len(numeric_cols))]
    for col in important_numeric_cols:
        df_with_features[f'{col}_squared'] = df_with_features[col] ** 2
        df_with_features[f'{col}_sqrt'] = np.sqrt(np.abs(df_with_features[col]))
    
    return df_with_features

def split_dataset(tables: Dict[str, pd.DataFrame], 
                 relationships: List[Tuple],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 temporal_split: bool = False,
                 temporal_column: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Divise le dataset en train/val/test
    
    Args:
        tables: Tables du dataset
        relationships: Relations entre tables
        train_ratio: Ratio d'entraînement
        val_ratio: Ratio de validation
        test_ratio: Ratio de test
        temporal_split: Si True, divise par temps
        temporal_column: Colonne temporelle pour le split temporel
    
    Returns:
        Dictionnaire avec splits train/val/test pour chaque table
    """
    
    # Valider les ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Les ratios doivent sommer à 1.0, obtenu: {total_ratio}")
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    if temporal_split and temporal_column:
        # Split temporel
        print("Split temporel du dataset...")
        
        # Trouver la table avec la colonne temporelle
        temporal_table = None
        for table_name, df in tables.items():
            if temporal_column in df.columns:
                temporal_table = table_name
                break
        
        if not temporal_table:
            raise ValueError(f"Colonne temporelle '{temporal_column}' non trouvée")
        
        # Trier par temps
        temporal_df = tables[temporal_table].copy()
        temporal_df = temporal_df.sort_values(temporal_column)
        
        # Calculer les indices de split
        n_total = len(temporal_df)
        train_end = int(n_total * train_ratio)
        val_end = train_end + int(n_total * val_ratio)
        
        # IDs pour chaque split
        train_ids = temporal_df.iloc[:train_end]['id'].values if 'id' in temporal_df.columns else None
        val_ids = temporal_df.iloc[train_end:val_end]['id'].values if 'id' in temporal_df.columns else None
        test_ids = temporal_df.iloc[val_end:]['id'].values if 'id' in temporal_df.columns else None
        
        # Diviser chaque table
        for table_name, df in tables.items():
            if 'id' in df.columns and train_ids is not None:
                # Split par ID
                splits['train'][table_name] = df[df['id'].isin(train_ids)].copy()
                splits['val'][table_name] = df[df['id'].isin(val_ids)].copy()
                splits['test'][table_name] = df[df['id'].isin(test_ids)].copy()
            else:
                # Split aléatoire
                splits['train'][table_name], splits['val'][table_name], splits['test'][table_name] = \
                    _random_split_table(df, train_ratio, val_ratio, test_ratio)
    
    else:
        # Split aléatoire
        print("Split aléatoire du dataset...")
        
        for table_name, df in tables.items():
            splits['train'][table_name], splits['val'][table_name], splits['test'][table_name] = \
                _random_split_table(df, train_ratio, val_ratio, test_ratio)
    
    # Afficher les statistiques
    print("\nStatistiques des splits:")
    for split_name in ['train', 'val', 'test']:
        total_rows = sum(len(df) for df in splits[split_name].values())
        print(f"  {split_name}: {total_rows} lignes totales")
        
        for table_name, df in splits[split_name].items():
            print(f"    - {table_name}: {len(df)} lignes")
    
    return splits

def _random_split_table(df: pd.DataFrame, train_ratio: float, 
                       val_ratio: float, test_ratio: float):
    """Divise une table aléatoirement"""
    
    # Mélanger
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)
    
    train_df = df_shuffled.iloc[:train_end].copy()
    val_df = df_shuffled.iloc[train_end:val_end].copy()
    test_df = df_shuffled.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def create_dataset_summary(tables: Dict[str, pd.DataFrame], 
                          relationships: List[Tuple]) -> pd.DataFrame:
    """
    Crée un résumé détaillé du dataset
    
    Args:
        tables: Tables du dataset
        relationships: Relations entre tables
    
    Returns:
        DataFrame avec le résumé
    """
    
    summary_data = []
    
    # Résumé par table
    for table_name, df in tables.items():
        summary_data.append({
            'type': 'table',
            'name': table_name,
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
            'temporal_cols': len(df.select_dtypes(include=['datetime']).columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 ** 2)
        })
    
    # Résumé des relations
    for i, rel in enumerate(relationships):
        if len(rel) >= 5:
            src_table, src_col, tgt_table, tgt_col, rel_type = rel[:5]
            summary_data.append({
                'type': 'relationship',
                'name': f'rel_{i}',
                'source_table': src_table,
                'source_column': src_col,
                'target_table': tgt_table,
                'target_column': tgt_col,
                'relationship_type': rel_type,
                'rows': '',
                'columns': '',
                'description': f'{src_table}.{src_col} → {tgt_table}.{tgt_col}'
            })
    
    # Créer le DataFrame de résumé
    summary_df = pd.DataFrame(summary_data)
    
    # Ajouter des statistiques globales
    total_rows = sum(len(df) for df in tables.values())
    total_columns = sum(len(df.columns) for df in tables.values())
    total_memory = sum(df.memory_usage(deep=True).sum() for df in tables.values()) / (1024 ** 2)
    
    global_stats = pd.DataFrame([{
        'type': 'global',
        'name': 'TOTAL',
        'rows': total_rows,
        'columns': total_columns,
        'numeric_cols': '',
        'categorical_cols': '',
        'temporal_cols': '',
        'missing_values': '',
        'missing_percentage': '',
        'memory_mb': total_memory,
        'description': f'{len(tables)} tables, {len(relationships)} relations'
    }])
    
    summary_df = pd.concat([summary_df, global_stats], ignore_index=True)
    
    return summary_df
