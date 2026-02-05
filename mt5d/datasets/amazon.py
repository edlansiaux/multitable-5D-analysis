"""
Loader pour Amazon Reviews dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import gzip
import warnings
warnings.filterwarnings('ignore')

from .base import MT5DDataset, DatasetMetadata, DatasetType

class AmazonReviewsLoader(MT5DDataset):
    """Chargeur pour Amazon Reviews dataset"""
    
    def __init__(self, data_dir: str, category: str = "All", 
                 sample_size: Optional[int] = None, download: bool = False):
        super().__init__(data_dir, download)
        self.category = category
        self.sample_size = sample_size
        self.is_loaded = False
    
    def download(self):
        """Télécharge Amazon Reviews dataset"""
        print("Amazon Reviews dataset nécessite un téléchargement manuel.")
        print("Disponible sur: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/")
        print("\nFormat recommandé:")
        print("1. Téléchargez les fichiers JSON.gz")
        print("2. Extrayez dans:", self.data_dir / "amazon_reviews")
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """Charge les tables Amazon Reviews"""
        
        if self.is_loaded:
            return self.tables
        
        amazon_dir = self.data_dir / "amazon_reviews"
        
        if not amazon_dir.exists():
            # Essayer de charger depuis un format alternatif
            amazon_dir = self.data_dir
        
        # Chercher les fichiers
        json_files = list(amazon_dir.glob("*.json.gz")) + list(amazon_dir.glob("*.json"))
        
        if not json_files:
            # Créer des données synthétiques pour la démo
            print("Aucun fichier Amazon trouvé. Création de données synthétiques...")
            self._create_synthetic_data()
            self.is_loaded = True
            return self.tables
        
        print(f"Chargement Amazon Reviews ({self.category})...")
        
        # Charger les reviews
        reviews_data = []
        for file_path in json_files[:2]:  # Limiter à 2 fichiers pour la performance
            try:
                if file_path.suffix == '.gz':
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        for line in f:
                            if self.sample_size and len(reviews_data) >= self.sample_size:
                                break
                            reviews_data.append(json.loads(line.strip()))
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if self.sample_size and len(reviews_data) >= self.sample_size:
                                break
                            reviews_data.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Erreur chargement {file_path}: {e}")
        
        # Convertir en DataFrame
        if reviews_data:
            reviews_df = pd.DataFrame(reviews_data)
            
            # Normaliser les colonnes
            column_mapping = {
                'reviewerID': 'user_id',
                'asin': 'product_id',
                'reviewText': 'review_text',
                'summary': 'review_summary',
                'overall': 'rating',
                'helpful': 'helpful_votes',
                'unixReviewTime': 'timestamp',
                'reviewTime': 'review_date'
            }
            
            # Renommer les colonnes existantes
            existing_columns = {k: v for k, v in column_mapping.items() if k in reviews_df.columns}
            reviews_df = reviews_df.rename(columns=existing_columns)
            
            # Ajouter les colonnes manquantes
            for col in column_mapping.values():
                if col not in reviews_df.columns:
                    reviews_df[col] = None
            
            # Sélectionner et ordonner les colonnes
            reviews_df = reviews_df[[
                'user_id', 'product_id', 'rating', 'review_text', 
                'review_summary', 'helpful_votes', 'timestamp', 'review_date'
            ]]
            
            self.tables['reviews'] = reviews_df
            print(f"  - Reviews: {len(reviews_df)} lignes")
        
        # Charger les metadata produits
        metadata_files = list(amazon_dir.glob("*_meta.json.gz")) + list(amazon_dir.glob("*_meta.json"))
        
        if metadata_files:
            metadata_data = []
            for file_path in metadata_files[:1]:  # Un seul fichier metadata
                try:
                    if file_path.suffix == '.gz':
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            for line in f:
                                metadata_data.append(json.loads(line.strip()))
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                metadata_data.append(json.loads(line.strip()))
                except Exception as e:
                    print(f"Erreur chargement metadata {file_path}: {e}")
            
            if metadata_data:
                metadata_df = pd.DataFrame(metadata_data)
                
                # Normaliser les colonnes
                meta_mapping = {
                    'asin': 'product_id',
                    'title': 'product_title',
                    'description': 'product_description',
                    'price': 'price',
                    'brand': 'brand',
                    'categories': 'categories',
                    'related': 'related_products'
                }
                
                existing_columns = {k: v for k, v in meta_mapping.items() if k in metadata_df.columns}
                metadata_df = metadata_df.rename(columns=existing_columns)
                
                # Ajouter les colonnes manquantes
                for col in meta_mapping.values():
                    if col not in metadata_df.columns:
                        metadata_df[col] = None
                
                # Sélectionner les colonnes
                metadata_df = metadata_df[[
                    'product_id', 'product_title', 'product_description',
                    'price', 'brand', 'categories', 'related_products'
                ]]
                
                self.tables['products'] = metadata_df
                print(f"  - Products: {len(metadata_df)} lignes")
        
        self.is_loaded = True
        return self.tables
    
    def _create_synthetic_data(self):
        """Crée des données synthétiques Amazon pour la démo"""
        
        print("Création de données Amazon synthétiques...")
        
        n_reviews = 10000
        n_products = 2000
        n_users = 5000
        
        # Reviews
        reviews_df = pd.DataFrame({
            'review_id': range(n_reviews),
            'user_id': np.random.choice(range(n_users), n_reviews),
            'product_id': np.random.choice(range(n_products), n_reviews),
            'rating': np.random.randint(1, 6, n_reviews),
            'review_text': [f"Review text {i}" for i in range(n_reviews)],
            'review_summary': [f"Summary {i}" for i in range(n_reviews)],
            'helpful_votes': np.random.randint(0, 50, n_reviews),
            'timestamp': pd.date_range('2020-01-01', periods=n_reviews, freq='H'),
            'verified_purchase': np.random.choice([True, False], n_reviews, p=[0.7, 0.3])
        })
        
        # Products
        categories = ['Electronics', 'Books', 'Home', 'Clothing', 'Sports']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
        
        products_df = pd.DataFrame({
            'product_id': range(n_products),
            'product_title': [f"Product {i}" for i in range(n_products)],
            'product_description': [f"Description for product {i}" for i in range(n_products)],
            'category': np.random.choice(categories, n_products),
            'subcategory': np.random.choice(['Sub1', 'Sub2', 'Sub3'], n_products),
            'brand': np.random.choice(brands + [None], n_products, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
            'price': np.random.uniform(5, 500, n_products).round(2),
            'avg_rating': np.random.uniform(1, 5, n_products),
            'review_count': np.random.randint(0, 100, n_products)
        })
        
        # Users
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'username': [f"user_{i}" for i in range(n_users)],
            'join_date': pd.date_range('2018-01-01', periods=n_users, freq='D'),
            'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_users),
            'helpful_votes_received': np.random.randint(0, 1000, n_users),
            'review_count': np.random.randint(0, 50, n_users)
        })
        
        self.tables['reviews'] = reviews_df
        self.tables['products'] = products_df
        self.tables['users'] = users_df
        
        print(f"  - Reviews synthétiques: {len(reviews_df)} lignes")
        print(f"  - Products synthétiques: {len(products_df)} lignes")
        print(f"  - Users synthétiques: {len(users_df)} lignes")
    
    def get_relationships(self) -> List[Tuple]:
        """Retourne les relations entre tables Amazon"""
        
        relationships = []
        
        if 'reviews' in self.tables and 'products' in self.tables:
            relationships.append(
                ('reviews', 'product_id', 'products', 'product_id', 'product_review')
            )
        
        if 'reviews' in self.tables and 'users' in self.tables:
            relationships.append(
                ('reviews', 'user_id', 'users', 'user_id', 'user_review')
            )
        
        # Relations produits-produits (via similarité)
        if 'products' in self.tables and 'related_products' in self.tables['products'].columns:
            relationships.append(
                ('products', 'product_id', 'products', 'related_products', 'similar_product')
            )
        
        return relationships
    
    def get_metadata(self) -> DatasetMetadata:
        """Retourne les métadonnées Amazon Reviews"""
        
        if not self.metadata:
            if not self.tables:
                self.load()
            
            self.metadata = DatasetMetadata(
                name="Amazon Reviews",
                type=DatasetType.ECOMMERCE,
                description="Amazon product reviews and metadata",
                source="University of California, San Diego",
                license="Academic use only",
                num_tables=len(self.tables),
                total_rows=sum(len(df) for df in self.tables.values()),
                total_columns=sum(len(df.columns) for df in self.tables.values()),
                has_temporal_data=True,
                has_relationships=True,
                download_url="https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/",
                citation="J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.",
                version="2.0"
            )
        
        return self.metadata
    
    def create_recommendation_task(self, task: str = "rating_prediction"):
        """Crée une tâche de recommandation"""
        
        if not self.is_loaded:
            self.load()
        
        if 'reviews' not in self.tables:
            raise ValueError("Table reviews nécessaire")
        
        reviews_df = self.tables['reviews'].copy()
        
        if task == "rating_prediction":
            # Prédiction de rating
            task_df = reviews_df[['user_id', 'product_id', 'rating']].copy()
            task_df.columns = ['user_id', 'item_id', 'rating']
            
            # Ajouter des features si disponibles
            if 'products' in self.tables:
                products_df = self.tables['products']
                if 'category' in products_df.columns:
                    task_df = pd.merge(
                        task_df,
                        products_df[['product_id', 'category']].rename(columns={'product_id': 'item_id'}),
                        on='item_id',
                        how='left'
                    )
            
            return task_df.dropna()
        
        elif task == "purchase_prediction":
            # Prédiction d'achat (classification binaire)
            task_df = reviews_df[['user_id', 'product_id']].copy()
            task_df.columns = ['user_id', 'item_id']
            
            # Marquer toutes les reviews comme achat (1)
            task_df['purchased'] = 1
            
            # Générer des négatifs (non-achats)
            all_users = task_df['user_id'].unique()
            all_items = task_df['item_id'].unique()
            
            # Échantillonner des paires user-item négatives
            n_negatives = len(task_df)
            negative_samples = []
            
            for _ in range(n_negatives):
                user = np.random.choice(all_users)
                item = np.random.choice(all_items)
                
                # Vérifier que la paire n'existe pas déjà
                if not ((task_df['user_id'] == user) & (task_df['item_id'] == item)).any():
                    negative_samples.append([user, item, 0])
            
            negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'purchased'])
            
            # Combiner positifs et négatifs
            task_df = pd.concat([task_df, negative_df], ignore_index=True)
            
            return task_df
        
        else:
            raise ValueError(f"Tâche non supportée: {task}")
    
    def create_sentiment_analysis_task(self):
        """Crée une tâche d'analyse de sentiment"""
        
        if not self.is_loaded:
            self.load()
        
        if 'reviews' not in self.tables:
            raise ValueError("Table reviews nécessaire")
        
        reviews_df = self.tables['reviews'].copy()
        
        # Créer des labels de sentiment basés sur le rating
        def rating_to_sentiment(rating):
            if rating >= 4:
                return 'positive'
            elif rating == 3:
                return 'neutral'
            else:
                return 'negative'
        
        reviews_df['sentiment'] = reviews_df['rating'].apply(rating_to_sentiment)
        
        # Retourner les données pour l'analyse de sentiment
        return reviews_df[['review_text', 'review_summary', 'sentiment', 'rating']].dropna()

# Enregistrer dans le registre
from .base import DatasetRegistry
DatasetRegistry.register("amazon-reviews", AmazonReviewsLoader)
