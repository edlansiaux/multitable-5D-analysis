#!/usr/bin/env python3
"""
Télécharge les données d'exemple pour MT5D.
"""
import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

EXAMPLE_DATA_URLS = {
    "medical_small": "https://example.com/data/medical_small.zip",
    "financial_small": "https://example.com/data/financial_small.zip",
    "ecommerce_small": "https://example.com/data/ecommerce_small.zip"
}

def create_synthetic_data(data_dir):
    """Crée des données synthétiques si pas de téléchargement"""
    print("Création de données synthétiques locales...")
    
    # Créer le répertoire
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Données médicales
    medical_dir = data_dir / "medical"
    medical_dir.mkdir(exist_ok=True)
    
    patients = pd.DataFrame({
        'patient_id': range(100),
        'age': np.random.randint(18, 90, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'diagnosis': np.random.choice(['I10', 'K37', 'M61.2', 'I41', 'I48'], 100)
    })
    
    labs = pd.DataFrame({
        'lab_id': range(500),
        'patient_id': np.random.choice(range(100), 500),
        'parameter': np.random.choice(['K+', 'Na+', 'Glucose', 'Creatinine'], 500),
        'value': np.random.normal(0, 1, 500),
        'timestamp': pd.date_range('2020-01-01', periods=500, freq='H')
    })
    
    patients.to_csv(medical_dir / "patients.csv", index=False)
    labs.to_csv(medical_dir / "labs.csv", index=False)
    
    # 2. Données financières
    financial_dir = data_dir / "financial"
    financial_dir.mkdir(exist_ok=True)
    
    customers = pd.DataFrame({
        'customer_id': range(50),
        'age': np.random.randint(18, 80, 50),
        'income': np.random.lognormal(10, 1, 50)
    })
    
    transactions = pd.DataFrame({
        'transaction_id': range(1000),
        'customer_id': np.random.choice(range(50), 1000),
        'amount': np.random.exponential(100, 1000),
        'is_fraud': np.random.binomial(1, 0.02, 1000)
    })
    
    customers.to_csv(financial_dir / "customers.csv", index=False)
    transactions.to_csv(financial_dir / "transactions.csv", index=False)
    
    # 3. Données e-commerce
    ecommerce_dir = data_dir / "ecommerce"
    ecommerce_dir.mkdir(exist_ok=True)
    
    users = pd.DataFrame({
        'user_id': range(50),
        'age': np.random.randint(18, 70, 50),
        'location': np.random.choice(['US', 'EU', 'ASIA'], 50)
    })
    
    products = pd.DataFrame({
        'product_id': range(100),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books'], 100),
        'price': np.random.uniform(10, 500, 100)
    })
    
    interactions = pd.DataFrame({
        'interaction_id': range(500),
        'user_id': np.random.choice(range(50), 500),
        'product_id': np.random.choice(range(100), 500),
        'rating': np.random.randint(1, 6, 500)
    })
    
    users.to_csv(ecommerce_dir / "users.csv", index=False)
    products.to_csv(ecommerce_dir / "products.csv", index=False)
    interactions.to_csv(ecommerce_dir / "interactions.csv", index=False)
    
    print(f"Données créées dans: {data_dir}")
    return True

def download_from_url(url, output_dir):
    """Télécharge depuis une URL (version simplifiée)"""
    print(f"Tentative de téléchargement depuis: {url}")
    
    # Dans la version réelle, on utiliserait requests
    # Pour l'exemple, on crée des données synthétiques
    return create_synthetic_data(output_dir)

def main():
    """Télécharge ou crée les données d'exemple"""
    print("Préparation des données d'exemple MT5D")
    print("=" * 50)
    
    # Répertoire cible
    data_dir = Path("example_data")
    
    if data_dir.exists():
        print(f"Le répertoire {data_dir} existe déjà.")
        response = input("Voulez-vous le regénérer? (y/n): ")
        if response.lower() != 'y':
            print("Abandon.")
            return
    
    # Essayer de télécharger, sinon créer localement
    success = False
    
    try:
        # Essayer le premier dataset
        for name, url in EXAMPLE_DATA_URLS.items():
            print(f"\nTentative de téléchargement: {name}")
            success = download_from_url(url, data_dir)
            if success:
                break
    except Exception as e:
        print(f"Erreur de téléchargement: {e}")
        print("Création de données synthétiques locales...")
        success = create_synthetic_data(data_dir)
    
    if success:
        print("\n" + "=" * 50)
        print("Données d'exemple prêtes!")
        print(f"Emplacement: {data_dir.absolute()}")
        print("\nStructure:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(str(data_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("Échec de la préparation des données.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
