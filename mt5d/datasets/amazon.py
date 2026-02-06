import pandas as pd
from typing import Dict, Tuple, List
from .base import BaseDataset

class AmazonDataset(BaseDataset):
    """
    Dataset Amazon Multi-Table (Section 7.1.2).
    Tables : Products, Reviews, Users, Categories.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        
    def load(self) -> Tuple[Dict[str, pd.DataFrame], List[Tuple]]:
        # Simulation du chargement (à remplacer par pd.read_csv réels)
        print(f"Chargement du dataset Amazon depuis {self.root_dir}...")
        
        tables = {
            'products': pd.DataFrame(columns=['asin', 'title', 'price']),
            'users': pd.DataFrame(columns=['reviewerID', 'name']),
            'reviews': pd.DataFrame(columns=['reviewerID', 'asin', 'reviewText', 'unixReviewTime']),
            'categories': pd.DataFrame(columns=['category_id', 'parent_id', 'name']),
            'product_categories': pd.DataFrame(columns=['asin', 'category_id'])
        }
        
        relationships = [
            # Reviews lie Users et Products (Many-to-Many résolu)
            ('users', 'reviewerID', 'reviews', 'reviewerID', 'one_to_many'),
            ('products', 'asin', 'reviews', 'asin', 'one_to_many'),
            
            # Hiérarchie des catégories (Self-referencing)
            ('categories', 'parent_id', 'categories', 'category_id', 'many_to_one'),
            
            # Produits <-> Catégories
            ('products', 'asin', 'product_categories', 'asin', 'one_to_many'),
            ('categories', 'category_id', 'product_categories', 'category_id', 'one_to_many')
        ]
        
        return tables, relationships

    def get_temporal_columns(self) -> Dict[str, str]:
        return {
            'reviews': 'unixReviewTime'
        }
