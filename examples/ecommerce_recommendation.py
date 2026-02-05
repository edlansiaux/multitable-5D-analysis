"""
Exemple: Système de recommandation e-commerce avec données multitable
"""
import pandas as pd
import numpy as np
from mt5d import MT5DPipeline

def generate_ecommerce_data(n_users=5000, n_products=1000, n_interactions=50000):
    """Génère des données e-commerce synthétiques"""
    
    # Utilisateurs
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(18, 70, n_users),
        'gender': np.random.choice(['M', 'F', 'O'], n_users),
        'location': np.random.choice(['US', 'EU', 'ASIA', 'OTHER'], n_users),
        'signup_date': pd.to_datetime(np.random.choice(
            pd.date_range('2020-01-01', '2022-01-01', freq='D'), n_users
        ))
    })
    
    # Produits
    products = pd.DataFrame({
        'product_id': range(n_products),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_products),
        'subcategory': np.random.choice(['Sub1', 'Sub2', 'Sub3', 'Sub4'], n_products),
        'price': np.random.uniform(5, 500, n_products),
        'brand': np.random.choice([f'Brand_{i}' for i in range(50)], n_products)
    })
    
    # Interactions (clics, ajouts au panier, achats)
    interaction_types = ['view', 'cart', 'purchase']
    interactions = pd.DataFrame({
        'interaction_id': range(n_interactions),
        'user_id': np.random.choice(users['user_id'], n_interactions),
        'product_id': np.random.choice(products['product_id'], n_interactions),
        'interaction_type': np.random.choice(interaction_types, n_interactions, p=[0.7, 0.2, 0.1]),
        'rating': np.random.randint(1, 6, n_interactions),
        'timestamp': pd.to_datetime(np.random.choice(
            pd.date_range('2022-01-01', '2023-01-01', freq='H'), n_interactions
        ))
    })
    
    # Sessions
    sessions = pd.DataFrame({
        'session_id': range(n_users * 3),  # ~3 sessions par utilisateur
        'user_id': np.random.choice(users['user_id'], n_users * 3),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_users * 3),
        'session_start': pd.to_datetime(np.random.choice(
            pd.date_range('2022-06-01', '2023-01-01', freq='H'), n_users * 3
        )),
        'session_duration_min': np.random.exponential(30, n_users * 3)
    })
    
    return {
        'users': users,
        'products': products,
        'interactions': interactions,
        'sessions': sessions
    }

def main():
    """Pipeline de recommandation e-commerce"""
    print("=== Système de Recommandation E-commerce ===")
    
    # 1. Générer les données
    print("1. Génération des données synthétiques...")
    tables = generate_ecommerce_data(
        n_users=1000,
        n_products=500,
        n_interactions=10000
    )
    
    print(f"   - Utilisateurs: {len(tables['users'])}")
    print(f"   - Produits: {len(tables['products'])}")
    print(f"   - Interactions: {len(tables['interactions'])}")
    print(f"   - Sessions: {len(tables['sessions'])}")
    
    # 2. Relations
    relationships = [
        ('interactions', 'user_id', 'users', 'user_id', 'user_interaction'),
        ('interactions', 'product_id', 'products', 'product_id', 'product_interaction'),
        ('sessions', 'user_id', 'users', 'user_id', 'user_session')
    ]
    
    # 3. Configurer le pipeline
    print("\n2. Configuration du pipeline de recommandation...")
    pipeline = MT5DPipeline()
    
    # Personnaliser la config pour la recommandation
    pipeline.config = {
        'models': {
            'recommendation': {
                'type': 'collaborative_filtering',
                'embedding_dim': 64,
                'negative_sampling': True
            }
        },
        'evaluation': {
            'metrics': ['precision_at_10', 'recall_at_10', 'ndcg'],
            'test_size': 0.2
        }
    }
    
    # 4. Exécuter
    print("\n3. Exécution du pipeline...")
    results = pipeline.run(
        tables=tables,
        relationships=relationships,
        target_task="product_recommendation"
    )
    
    # 5. Résultats
    print("\n4. Résultats de recommandation:")
    
    if 'recommendations' in results['results']:
        # Exemple: recommandations pour les premiers utilisateurs
        user_recommendations = results['results']['recommendations']
        
        print(f"\n5. Exemples de recommandations:")
        for user_id in list(user_recommendations.keys())[:3]:
            recs = user_recommendations[user_id]
            print(f"   Utilisateur {user_id}: {', '.join(map(str, recs[:5]))}")
    
    # Évaluation
    if 'evaluation' in results and 'recommendation_metrics' in results['evaluation']:
        eval_metrics = results['evaluation']['recommendation_metrics']
        print(f"\n6. Métriques d'évaluation:")
        for metric, value in eval_metrics.items():
            print(f"   - {metric}: {value:.4f}")
    
    # Insights
    if 'insights' in results['results']:
        print(f"\n7. Insights découverts:")
        for i, insight in enumerate(results['results']['insights'][:5], 1):
            print(f"   {i}. {insight}")
    
    return results

if __name__ == "__main__":
    main()
