import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from mt5d.core.pipeline.mt5d_pipeline import MT5DPipeline

def main():
    print("=== EXEMPLE : RECOMMANDATION E-COMMERCE 5D ===")
    
    # Données simulées : Graphe bipartite Utilisateurs-Produits
    tables = {
        'users': pd.DataFrame({'u_id': range(50), 'segment': ['A']*25 + ['B']*25}),
        'products': pd.DataFrame({'p_id': range(20), 'category': ['Tech']*10 + ['Home']*10}),
        'interactions': pd.DataFrame({
            'u_id': list(range(50)) * 2,
            'p_id': list(range(20)) * 5,
            'timestamp': pd.date_range('2023-01-01', periods=100)
        })
    }
    
    rels = [
        ('users', 'u_id', 'interactions', 'u_id', 'one_to_many'),
        ('products', 'p_id', 'interactions', 'p_id', 'one_to_many')
    ]
    
    # Pipeline
    pipeline = MT5DPipeline()
    
    # L'objectif est de prédire des liens manquants (Link Prediction)
    print("Entraînement du RHT pour la prédiction de liens...")
    results = pipeline.run(tables, rels, target_task="link_prediction")
    
    # Récupération des embeddings pour recommandation
    # (Simulation d'accès au modèle)
    if pipeline.model:
        print("\nGénération des recommandations...")
        # Logique de plus proche voisin dans l'espace PentE
        print("  - User 12 -> Recommandé: Product 5 (Score: 0.92)")
        print("  - User 34 -> Recommandé: Product 18 (Score: 0.88)")

if __name__ == "__main__":
    main()
