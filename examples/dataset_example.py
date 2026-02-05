"""
Exemple d'utilisation des datasets MT5D
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from mt5d.datasets import (
    DatasetRegistry,
    MIMICLoader,
    AmazonReviewsLoader,
    FinancialTransactionsLoader,
    SyntheticDataGenerator,
    create_dataset_summary,
    preprocess_dataset,
    split_dataset
)

def demonstrate_datasets():
    """Démontre l'utilisation de différents datasets"""
    
    print("=" * 80)
    print("DÉMONSTRATION DES DATASETS MT5D")
    print("=" * 80)
    
    data_dir = "./data"
    Path(data_dir).mkdir(exist_ok=True)
    
    # 1. Lister tous les datasets disponibles
    print("\n1. Datasets disponibles:")
    datasets = DatasetRegistry.list_datasets()
    for i, dataset_name in enumerate(datasets, 1):
        print(f"   {i}. {dataset_name}")
    
    # 2. Charger un dataset synthétique (pas de téléchargement nécessaire)
    print("\n2. Chargement d'un dataset synthétique médical...")
    
    medical_dataset = DatasetRegistry.get_dataset(
        "synthetic-medical",
        data_dir=data_dir,
        scale="medium"
    )
    
    # Charger les données
    tables = medical_dataset.load()
    
    # Obtenir les relations
    relationships = medical_dataset.get_relationships()
    
    # Obtenir les métadonnées
    metadata = medical_dataset.get_metadata()
    
    print(f"   - Nom: {metadata.name}")
    print(f"   - Type: {metadata.type.value}")
    print(f"   - Tables: {metadata.num_tables}")
    print(f"   - Lignes totales: {metadata.total_rows:,}")
    print(f"   - Relations: {len(relationships)}")
    
    # 3. Créer un résumé du dataset
    print("\n3. Résumé du dataset médical:")
    summary = create_dataset_summary(tables, relationships)
    print(summary[['type', 'name', 'rows', 'columns', 'memory_mb']].to_string())
    
    # 4. Prétraiter les données
    print("\n4. Prétraitement des données...")
    
    preprocessing_steps = {
        'patients': ['fill_missing', 'encode_categorical'],
        'visits': ['fill_missing', 'extract_temporal_features'],
        'diagnoses': ['fill_missing', 'encode_categorical'],
        'medications': ['fill_missing'],
        'lab_results': ['fill_missing', 'remove_outliers']
    }
    
    processed_tables = preprocess_dataset(tables, preprocessing_steps)
    
    print("   Tables prétraitées:")
    for table_name, df in processed_tables.items():
        print(f"   - {table_name}: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # 5. Diviser le dataset
    print("\n5. Division du dataset (train/val/test)...")
    
    splits = split_dataset(
        tables=processed_tables,
        relationships=relationships,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        temporal_split=False
    )
    
    print("   Tailles des splits:")
    for split_name in ['train', 'val', 'test']:
        total_rows = sum(len(df) for df in splits[split_name].values())
        print(f"   - {split_name}: {total_rows:,} lignes")
    
    # 6. Visualiser les données
    print("\n6. Visualisation des données...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distribution d'âge des patients
    if 'patients' in tables and 'age' in tables['patients'].columns:
        axes[0, 0].hist(tables['patients']['age'].dropna(), bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution d\'âge des patients')
        axes[0, 0].set_xlabel('Âge')
        axes[0, 0].set_ylabel('Fréquence')
    
    # Plot 2: Types de visites
    if 'visits' in tables and 'visit_type' in tables['visits'].columns:
        visit_counts = tables['visits']['visit_type'].value_counts()
        axes[0, 1].bar(visit_counts.index, visit_counts.values, color='lightcoral')
        axes[0, 1].set_title('Types de visites')
        axes[0, 1].set_xlabel('Type de visite')
        axes[0, 1].set_ylabel('Nombre')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Tendance temporelle des visites
    if 'visits' in tables and 'visit_date' in tables['visits'].columns:
        visits_df = tables['visits'].copy()
        visits_df['visit_date'] = pd.to_datetime(visits_df['visit_date'])
        visits_by_month = visits_df.set_index('visit_date').resample('M').size()
        
        axes[1, 0].plot(visits_by_month.index, visits_by_month.values, 
                       marker='o', color='green', linewidth=2)
        axes[1, 0].set_title('Visites par mois')
        axes[1, 0].set_xlabel('Mois')
        axes[1, 0].set_ylabel('Nombre de visites')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Distribution des diagnostiques
    if 'diagnoses' in tables and 'severity' in tables['diagnoses'].columns:
        severity_counts = tables['diagnoses']['severity'].value_counts()
        axes[1, 1].pie(severity_counts.values, labels=severity_counts.index,
                      autopct='%1.1f%%', colors=['gold', 'orange', 'red'])
        axes[1, 1].set_title('Sévérité des diagnostiques')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=300, bbox_inches='tight')
    print("   ✓ Visualisations sauvegardées: dataset_visualization.png")
    
    # 7. Créer une tâche de prédiction
    print("\n7. Création d'une tâche de prédiction...")
    
    # Exemple: prédire la durée de séjour
    if 'visits' in processed_tables:
        visits_df = processed_tables['visits'].copy()
        
        # Créer une variable cible (exemple simplifié)
        # Dans la réalité, cela nécessiterait plus de feature engineering
        if 'systolic_bp' in visits_df.columns and 'diastolic_bp' in visits_df.columns:
            visits_df['blood_pressure_risk'] = (
                (visits_df['systolic_bp'] > 140) | 
                (visits_df['diastolic_bp'] > 90)
            ).astype(int)
            
            # Features
            feature_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature']
            feature_cols = [col for col in feature_cols if col in visits_df.columns]
            
            X = visits_df[feature_cols].copy()
            y = visits_df['blood_pressure_risk'].copy()
            
            print(f"   - Tâche: Prédiction de risque d'hypertension")
            print(f"   - Features: {len(feature_cols)}")
            print(f"   - Échantillons: {len(X)}")
            print(f"   - Classe positive: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # 8. Sauvegarder le dataset
    print("\n8. Sauvegarde du dataset...")
    
    medical_dataset.save(
        output_dir="./saved_medical_dataset",
        format="parquet"
    )
    
    print("   ✓ Dataset sauvegardé dans: ./saved_medical_dataset/")
    
    # 9. Charger un autre type de dataset
    print("\n9. Chargement d'un dataset e-commerce...")
    
    try:
        ecommerce_dataset = DatasetRegistry.get_dataset(
            "synthetic-ecommerce",
            data_dir=data_dir,
            scale="small"
        )
        
        ecommerce_tables = ecommerce_dataset.load()
        ecommerce_metadata = ecommerce_dataset.get_metadata()
        
        print(f"   - Nom: {ecommerce_metadata.name}")
        print(f"   - Tables: {len(ecommerce_tables)}")
        
        # Afficher les statistiques des produits
        if 'products' in ecommerce_tables:
            products_df = ecommerce_tables['products']
            print(f"   - Produits: {len(products_df)}")
            print(f"   - Catégories uniques: {products_df['category'].nunique()}")
            print(f"   - Prix moyen: ${products_df['price'].mean():.2f}")
    
    except Exception as e:
        print(f"   ⚠ Erreur chargement e-commerce: {e}")
    
    # 10. Benchmark de différents datasets
    print("\n10. Benchmark de différents datasets:")
    
    dataset_configs = [
        ("synthetic-medical", "medium"),
        ("synthetic-ecommerce", "small"),
        ("synthetic-financial", "small")
    ]
    
    benchmark_results = []
    
    for dataset_name, scale in dataset_configs:
        try:
            dataset = DatasetRegistry.get_dataset(
                dataset_name,
                data_dir=data_dir,
                scale=scale
            )
            
            tables = dataset.load()
            metadata = dataset.get_metadata()
            
            benchmark_results.append({
                'dataset': dataset_name,
                'tables': len(tables),
                'total_rows': metadata.total_rows,
                'total_columns': metadata.total_columns,
                'memory_est_mb': sum(
                    df.memory_usage(deep=True).sum() / (1024 ** 2) 
                    for df in tables.values()
                )
            })
            
        except Exception as e:
            print(f"   ⚠ Erreur benchmark {dataset_name}: {e}")
    
    if benchmark_results:
        benchmark_df = pd.DataFrame(benchmark_results)
        print("\n" + benchmark_df.to_string())
    
    print("\n" + "=" * 80)
    print("DÉMONSTRATION TERMINÉE!")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_datasets()
