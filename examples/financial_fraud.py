"""
Exemple: Détection de fraude financière avec données multitable
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mt5d import MT5DPipeline

def generate_financial_data(n_customers=1000, n_transactions=10000):
    """Génère des données financières synthétiques"""
    
    # Clients
    customers = pd.DataFrame({
        'customer_id': range(n_customers),
        'age': np.random.randint(18, 80, n_customers),
        'income': np.random.lognormal(10, 1, n_customers),
        'risk_score': np.random.beta(2, 5, n_customers),
        'registration_date': pd.to_datetime(np.random.choice(
            pd.date_range('2018-01-01', '2020-01-01', freq='D'), n_customers
        ))
    })
    
    # Comptes
    accounts = pd.DataFrame({
        'account_id': range(n_customers * 2),  # 2 comptes par client en moyenne
        'customer_id': np.random.choice(range(n_customers), n_customers * 2),
        'account_type': np.random.choice(['checking', 'savings', 'business'], n_customers * 2),
        'balance': np.random.exponential(5000, n_customers * 2),
        'open_date': pd.to_datetime(np.random.choice(
            pd.date_range('2019-01-01', '2021-01-01', freq='D'), n_customers * 2
        ))
    })
    
    # Transactions
    base_date = datetime(2021, 1, 1)
    transactions = pd.DataFrame({
        'transaction_id': range(n_transactions),
        'account_id': np.random.choice(accounts['account_id'], n_transactions),
        'amount': np.random.exponential(200, n_transactions),
        'merchant_category': np.random.choice(['retail', 'restaurant', 'travel', 'online'], n_transactions),
        'is_fraud': np.random.binomial(1, 0.01, n_transactions),  # 1% de fraude
        'timestamp': [
            base_date + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            ) for _ in range(n_transactions)
        ]
    })
    
    # Ajouter quelques patterns de fraude
    fraud_indices = transactions[transactions['is_fraud'] == 1].index
    if len(fraud_indices) > 0:
        # Les fraudes ont souvent des montants élevés
        transactions.loc[fraud_indices, 'amount'] *= np.random.uniform(5, 20, len(fraud_indices))
        # Et ont lieu la nuit
        for idx in fraud_indices:
            transactions.loc[idx, 'timestamp'] = transactions.loc[idx, 'timestamp'].replace(
                hour=np.random.choice([1, 2, 3, 4])
            )
    
    return {
        'customers': customers,
        'accounts': accounts,
        'transactions': transactions
    }

def main():
    """Pipeline de détection de fraude"""
    print("=== Détection de Fraude Financière ===")
    
    # 1. Générer ou charger les données
    print("1. Chargement des données...")
    tables = generate_financial_data()
    
    print(f"   - Clients: {len(tables['customers'])}")
    print(f"   - Comptes: {len(tables['accounts'])}")
    print(f"   - Transactions: {len(tables['transactions'])}")
    print(f"   - Transactions frauduleuses: {tables['transactions']['is_fraud'].sum()}")
    
    # 2. Définir les relations
    relationships = [
        ('accounts', 'customer_id', 'customers', 'customer_id', 'customer_account'),
        ('transactions', 'account_id', 'accounts', 'account_id', 'account_transaction')
    ]
    
    # 3. Configurer le pipeline
    print("\n2. Configuration du pipeline...")
    pipeline = MT5DPipeline(config_path="configs/financial_config.yaml")
    
    # 4. Exécuter le pipeline
    print("\n3. Exécution du pipeline...")
    results = pipeline.run(
        tables=tables,
        relationships=relationships,
        target_task="fraud_detection"
    )
    
    # 5. Analyser les résultats
    print("\n4. Résultats:")
    
    metrics = results['metrics']
    print(f"   - Volume total: {metrics.volume['total_rows']:,} lignes")
    print(f"   - Mémoire: {metrics.volume['total_memory_mb']:.1f} MB")
    
    # Insights sur les fraudes détectées
    if 'fraud_patterns' in results['results']:
        patterns = results['results']['fraud_patterns']
        print(f"\n5. Patterns de fraude détectés:")
        for i, pattern in enumerate(patterns[:5], 1):
            print(f"   {i}. {pattern}")
    
    # Évaluation
    if 'evaluation' in results and 'fraud_metrics' in results['evaluation']:
        eval_metrics = results['evaluation']['fraud_metrics']
        print(f"\n6. Performance de détection:")
        print(f"   - Précision: {eval_metrics.get('precision', 0):.3f}")
        print(f"   - Rappel: {eval_metrics.get('recall', 0):.3f}")
        print(f"   - F1-score: {eval_metrics.get('f1', 0):.3f}")
        print(f"   - AUC-ROC: {eval_metrics.get('roc_auc', 0):.3f}")
    
    # 6. Sauvegarder les résultats
    print("\n7. Sauvegarde des résultats...")
    pipeline.save_pipeline("outputs/financial_fraud_analysis")
    print("   Résultats sauvegardés dans: outputs/financial_fraud_analysis")
    
    return results

if __name__ == "__main__":
    main()
