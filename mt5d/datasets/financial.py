"""
Loader pour les données financières
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .base import MT5DDataset, DatasetMetadata, DatasetType

class FinancialTransactionsLoader(MT5DDataset):
    """Chargeur pour les données de transactions financières"""
    
    def __init__(self, data_dir: str, dataset: str = "synthetic", 
                 include_fraud: bool = True, download: bool = False):
        super().__init__(data_dir, download)
        self.dataset = dataset  # "synthetic", "kaggle", "bank"
        self.include_fraud = include_fraud
        self.is_loaded = False
    
    def download(self):
        """Télécharge des données financières publiques"""
        
        if self.dataset == "kaggle":
            print("Kaggle Credit Card Fraud dataset:")
            print("Disponible sur: https://www.kaggle.com/mlg-ulb/creditcardfraud")
            print("\nTéléchargez et extrayez dans:", self.data_dir / "creditcardfraud")
        
        elif self.dataset == "bank":
            print("Bank Marketing dataset (UCI):")
            print("Disponible sur: https://archive.ics.uci.edu/ml/datasets/bank+marketing")
            print("\nTéléchargez et extrayez dans:", self.data_dir / "bank_marketing")
        
        else:
            print("Données financières synthétiques seront générées automatiquement.")
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """Charge les tables financières"""
        
        if self.is_loaded:
            return self.tables
        
        if self.dataset == "kaggle":
            self._load_kaggle_creditcard()
        elif self.dataset == "bank":
            self._load_bank_marketing()
        else:
            # Par défaut: données synthétiques
            self._create_synthetic_data()
        
        self.is_loaded = True
        return self.tables
    
    def _load_kaggle_creditcard(self):
        """Charge Kaggle Credit Card Fraud dataset"""
        
        data_dir = self.data_dir / "creditcardfraud"
        
        if not data_dir.exists():
            print("Dataset Kaggle non trouvé. Création de données synthétiques...")
            self._create_synthetic_data()
            return
        
        print("Chargement Kaggle Credit Card Fraud...")
        
        try:
            # Transactions
            transactions_path = data_dir / "creditcard.csv"
            if transactions_path.exists():
                transactions_df = pd.read_csv(transactions_path)
                
                # Renommer les colonnes
                column_mapping = {
                    'Time': 'timestamp',
                    'V1': 'feature_1', 'V2': 'feature_2', 'V3': 'feature_3',
                    'V4': 'feature_4', 'V5': 'feature_5', 'V6': 'feature_6',
                    'V7': 'feature_7', 'V8': 'feature_8', 'V9': 'feature_9',
                    'V10': 'feature_10', 'V11': 'feature_11', 'V12': 'feature_12',
                    'V13': 'feature_13', 'V14': 'feature_14', 'V15': 'feature_15',
                    'V16': 'feature_16', 'V17': 'feature_17', 'V18': 'feature_18',
                    'V19': 'feature_19', 'V20': 'feature_20', 'V21': 'feature_21',
                    'V22': 'feature_22', 'V23': 'feature_23', 'V24': 'feature_24',
                    'V25': 'feature_25', 'V26': 'feature_26', 'V27': 'feature_27',
                    'V28': 'feature_28',
                    'Amount': 'amount',
                    'Class': 'is_fraud'
                }
                
                # Renommer les colonnes existantes
                existing_columns = {k: v for k, v in column_mapping.items() if k in transactions_df.columns}
                transactions_df = transactions_df.rename(columns=existing_columns)
                
                # Ajouter un ID de transaction
                transactions_df['transaction_id'] = range(len(transactions_df))
                
                # Ajouter des colonnes manquantes
                if 'customer_id' not in transactions_df.columns:
                    transactions_df['customer_id'] = np.random.randint(1, 10000, len(transactions_df))
                
                if 'merchant_id' not in transactions_df.columns:
                    transactions_df['merchant_id'] = np.random.randint(1, 500, len(transactions_df))
                
                self.tables['transactions'] = transactions_df
                print(f"  - Transactions: {len(transactions_df)} lignes")
        
        except Exception as e:
            print(f"Erreur chargement Kaggle dataset: {e}")
            print("Création de données synthétiques...")
            self._create_synthetic_data()
    
    def _load_bank_marketing(self):
        """Charge Bank Marketing dataset (UCI)"""
        
        data_dir = self.data_dir / "bank_marketing"
        
        if not data_dir.exists():
            print("Dataset Bank Marketing non trouvé. Création de données synthétiques...")
            self._create_synthetic_data()
            return
        
        print("Chargement Bank Marketing...")
        
        try:
            # Chercher le fichier CSV
            csv_files = list(data_dir.glob("*.csv"))
            
            if csv_files:
                bank_df = pd.read_csv(csv_files[0], sep=';')
                
                # Renommer les colonnes
                bank_df.columns = [col.strip('"') for col in bank_df.columns]
                
                # Créer différentes tables
                # Clients
                clients_df = bank_df[['age', 'job', 'marital', 'education', 
                                    'default', 'housing', 'loan']].copy()
                clients_df['client_id'] = range(len(clients_df))
                clients_df = clients_df.drop_duplicates().reset_index(drop=True)
                
                # Contacts
                contacts_df = bank_df[['contact', 'month', 'day_of_week', 
                                      'duration', 'campaign', 'pdays', 
                                      'previous', 'poutcome']].copy()
                contacts_df['contact_id'] = range(len(contacts_df))
                contacts_df['client_id'] = np.random.choice(
                    clients_df['client_id'], len(contacts_df)
                )
                
                # Résultats
                results_df = bank_df[['y']].copy()
                results_df['result_id'] = range(len(results_df))
                results_df['contact_id'] = np.random.choice(
                    contacts_df['contact_id'], len(results_df)
                )
                results_df['success'] = results_df['y'].map({'yes': 1, 'no': 0})
                
                self.tables['clients'] = clients_df
                self.tables['contacts'] = contacts_df
                self.tables['results'] = results_df
                
                print(f"  - Clients: {len(clients_df)} lignes")
                print(f"  - Contacts: {len(contacts_df)} lignes")
                print(f"  - Results: {len(results_df)} lignes")
        
        except Exception as e:
            print(f"Erreur chargement Bank Marketing: {e}")
            print("Création de données synthétiques...")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Crée des données financières synthétiques"""
        
        print("Création de données financières synthétiques...")
        
        n_customers = 5000
        n_merchants = 200
        n_transactions = 50000
        n_accounts = 7000
        
        # Customers
        customers_df = pd.DataFrame({
            'customer_id': range(n_customers),
            'age': np.random.randint(18, 80, n_customers),
            'income': np.random.lognormal(10, 1, n_customers).round(2),
            'credit_score': np.random.randint(300, 850, n_customers),
            'employment_status': np.random.choice(
                ['employed', 'self-employed', 'unemployed', 'retired'], 
                n_customers,
                p=[0.6, 0.2, 0.1, 0.1]
            ),
            'years_at_address': np.random.exponential(5, n_customers).astype(int),
            'registration_date': pd.to_datetime(
                np.random.choice(pd.date_range('2015-01-01', '2022-12-31', freq='D'), 
                               n_customers)
            )
        })
        
        # Accounts
        accounts_df = pd.DataFrame({
            'account_id': range(n_accounts),
            'customer_id': np.random.choice(customers_df['customer_id'], n_accounts),
            'account_type': np.random.choice(['checking', 'savings', 'investment'], 
                                           n_accounts, p=[0.5, 0.3, 0.2]),
            'balance': np.random.exponential(5000, n_accounts).round(2),
            'open_date': pd.to_datetime(
                np.random.choice(pd.date_range('2016-01-01', '2023-06-30', freq='D'), 
                               n_accounts)
            ),
            'status': np.random.choice(['active', 'inactive', 'closed'], 
                                     n_accounts, p=[0.8, 0.1, 0.1])
        })
        
        # Merchants
        merchant_categories = ['Retail', 'Restaurant', 'Travel', 'Online', 
                             'Entertainment', 'Services', 'Healthcare']
        
        merchants_df = pd.DataFrame({
            'merchant_id': range(n_merchants),
            'merchant_name': [f"Merchant_{i}" for i in range(n_merchants)],
            'category': np.random.choice(merchant_categories, n_merchants),
            'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'FR', 'DE'], n_merchants),
            'risk_score': np.random.beta(2, 5, n_merchants)  # 0-1, plus haut = plus risqué
        })
        
        # Transactions
        base_date = pd.Timestamp('2023-01-01')
        transactions_df = pd.DataFrame({
            'transaction_id': range(n_transactions),
            'account_id': np.random.choice(accounts_df['account_id'], n_transactions),
            'merchant_id': np.random.choice(merchants_df['merchant_id'], n_transactions),
            'amount': np.random.exponential(200, n_transactions).round(2),
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_transactions, 
                                       p=[0.7, 0.2, 0.1]),
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], 
                                               n_transactions, p=[0.8, 0.15, 0.05]),
            'timestamp': [
                base_date + pd.Timedelta(
                    days=np.random.randint(0, 365),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                    seconds=np.random.randint(0, 60)
                ) for _ in range(n_transactions)
            ]
        })
        
        # Ajouter des fraudes
        if self.include_fraud:
            n_fraud = int(n_transactions * 0.01)  # 1% de fraudes
            fraud_indices = np.random.choice(range(n_transactions), n_fraud, replace=False)
            
            # Marquer comme fraude
            transactions_df['is_fraud'] = 0
            transactions_df.loc[fraud_indices, 'is_fraud'] = 1
            
            # Caractéristiques des fraudes
            # Montants plus élevés
            transactions_df.loc[fraud_indices, 'amount'] *= np.random.uniform(5, 20, n_fraud)
            
            # Heures nocturnes
            for idx in fraud_indices:
                transactions_df.loc[idx, 'timestamp'] = transactions_df.loc[idx, 'timestamp'].replace(
                    hour=np.random.choice([1, 2, 3, 4])
                )
            
            # Catégories à risque
            high_risk_merchants = merchants_df[merchants_df['risk_score'] > 0.8]['merchant_id'].values
            transactions_df.loc[fraud_indices, 'merchant_id'] = np.random.choice(
                high_risk_merchants, 
                size=min(n_fraud, len(high_risk_merchants)),
                replace=True
            )
        
        # Loans
        n_loans = 1000
        loans_df = pd.DataFrame({
            'loan_id': range(n_loans),
            'customer_id': np.random.choice(customers_df['customer_id'], n_loans),
            'amount': np.random.uniform(1000, 50000, n_loans).round(2),
            'interest_rate': np.random.uniform(0.02, 0.15, n_loans),
            'term_months': np.random.choice([12, 24, 36, 48, 60], n_loans),
            'start_date': pd.to_datetime(
                np.random.choice(pd.date_range('2020-01-01', '2023-06-30', freq='D'), 
                               n_loans)
            ),
            'status': np.random.choice(['active', 'paid', 'defaulted'], 
                                     n_loans, p=[0.6, 0.35, 0.05])
        })
        
        self.tables['customers'] = customers_df
        self.tables['accounts'] = accounts_df
        self.tables['merchants'] = merchants_df
        self.tables['transactions'] = transactions_df
        self.tables['loans'] = loans_df
        
        print(f"  - Customers: {len(customers_df)} lignes")
        print(f"  - Accounts: {len(accounts_df)} lignes")
        print(f"  - Merchants: {len(merchants_df)} lignes")
        print(f"  - Transactions: {len(transactions_df)} lignes")
        print(f"  - Loans: {len(loans_df)} lignes")
        
        if self.include_fraud:
            fraud_count = transactions_df['is_fraud'].sum()
            print(f"  - Transactions frauduleuses: {fraud_count} ({fraud_count/n_transactions*100:.2f}%)")
    
    def get_relationships(self) -> List[Tuple]:
        """Retourne les relations entre tables financières"""
        
        relationships = []
        
        if 'accounts' in self.tables and 'customers' in self.tables:
            relationships.append(
                ('accounts', 'customer_id', 'customers', 'customer_id', 'customer_account')
            )
        
        if 'transactions' in self.tables and 'accounts' in self.tables:
            relationships.append(
                ('transactions', 'account_id', 'accounts', 'account_id', 'account_transaction')
            )
        
        if 'transactions' in self.tables and 'merchants' in self.tables:
            relationships.append(
                ('transactions', 'merchant_id', 'merchants', 'merchant_id', 'merchant_transaction')
            )
        
        if 'loans' in self.tables and 'customers' in self.tables:
            relationships.append(
                ('loans', 'customer_id', 'customers', 'customer_id', 'customer_loan')
            )
        
        return relationships
    
    def get_metadata(self) -> DatasetMetadata:
        """Retourne les métadonnées financières"""
        
        if not self.metadata:
            if not self.tables:
                self.load()
            
            self.metadata = DatasetMetadata(
                name="Financial Transactions",
                type=DatasetType.FINANCIAL,
                description="Synthetic financial transactions dataset with customers, accounts, merchants, and fraud detection",
                source="Synthetic / Kaggle / UCI",
                license="MIT",
                num_tables=len(self.tables),
                total_rows=sum(len(df) for df in self.tables.values()),
                total_columns=sum(len(df.columns) for df in self.tables.values()),
                has_temporal_data=True,
                has_relationships=True,
                download_url="Synthetic" if self.dataset == "synthetic" else None,
                citation="Synthetic data for demonstration purposes",
                version="1.0"
            )
        
        return self.metadata
    
    def create_fraud_detection_task(self):
        """Crée une tâche de détection de fraude"""
        
        if not self.is_loaded:
            self.load()
        
        if 'transactions' not in self.tables:
            raise ValueError("Table transactions nécessaire")
        
        transactions_df = self.tables['transactions'].copy()
        
        if 'is_fraud' not in transactions_df.columns:
            # Ajouter une colonne de fraude synthétique
            transactions_df['is_fraud'] = 0
            n_fraud = int(len(transactions_df) * 0.01)
            fraud_indices = np.random.choice(transactions_df.index, n_fraud, replace=False)
            transactions_df.loc[fraud_indices, 'is_fraud'] = 1
        
        # Préparer les features
        features = ['amount', 'transaction_type']
        
        # Ajouter des features temporelles
        if 'timestamp' in transactions_df.columns:
            transactions_df['hour'] = pd.to_datetime(transactions_df['timestamp']).dt.hour
            transactions_df['day_of_week'] = pd.to_datetime(transactions_df['timestamp']).dt.dayofweek
            transactions_df['is_night'] = transactions_df['hour'].between(0, 5).astype(int)
            
            features.extend(['hour', 'day_of_week', 'is_night'])
        
        # Ajouter des features de merchant
        if 'merchants' in self.tables and 'merchant_id' in transactions_df.columns:
            merchant_features = self.tables['merchants'][['merchant_id', 'risk_score']]
            transactions_df = pd.merge(
                transactions_df,
                merchant_features,
                on='merchant_id',
                how='left'
            )
            
            if 'risk_score' in transactions_df.columns:
                features.append('risk_score')
        
        # Retourner les données pour la détection de fraude
        X = transactions_df[features].copy()
        y = transactions_df['is_fraud'].copy()
        
        return X, y
    
    def create_credit_scoring_task(self):
        """Crée une tâche de scoring de crédit"""
        
        if not self.is_loaded:
            self.load()
        
        if 'customers' not in self.tables:
            raise ValueError("Table customers nécessaire")
        
        customers_df = self.tables['customers'].copy()
        
        # Créer une variable cible basée sur le credit_score
        customers_df['credit_risk'] = pd.cut(
            customers_df['credit_score'],
            bins=[0, 579, 669, 739, 799, 850],
            labels=['poor', 'fair', 'good', 'very_good', 'excellent']
        )
        
        # Features
        features = ['age', 'income', 'employment_status', 'years_at_address']
        
        # Encoder les variables catégorielles
        X = pd.get_dummies(customers_df[features], drop_first=True)
        y = customers_df['credit_risk']
        
        return X, y
    
    def create_customer_segmentation_task(self):
        """Crée une tâche de segmentation client"""
        
        if not self.is_loaded:
            self.load()
        
        if 'customers' not in self.tables or 'transactions' not in self.tables:
            raise ValueError("Tables customers et transactions nécessaires")
        
        customers_df = self.tables['customers'].copy()
        transactions_df = self.tables['transactions'].copy()
        
        # Agréger les transactions par client
        customer_stats = transactions_df.groupby('account_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        # Renommer les colonnes
        customer_stats.columns = ['account_id', 'tx_count', 'tx_total', 'tx_mean', 'tx_std', 
                                 'first_tx', 'last_tx']
        
        # Calculer la fréquence et récence
        customer_stats['tx_frequency'] = customer_stats['tx_count']
        customer_stats['recency_days'] = (
            pd.Timestamp.now() - pd.to_datetime(customer_stats['last_tx'])
        ).dt.days
        
        # Fusionner avec les données clients
        if 'accounts' in self.tables:
            accounts_df = self.tables['accounts'].copy()
            customer_stats = pd.merge(
                customer_stats,
                accounts_df[['account_id', 'customer_id', 'balance']],
                on='account_id',
                how='left'
            )
            
            segmentation_df = pd.merge(
                customers_df,
                customer_stats.groupby('customer_id').agg({
                    'tx_count': 'sum',
                    'tx_total': 'sum',
                    'balance': 'mean',
                    'recency_days': 'min'
                }).reset_index(),
                on='customer_id',
                how='left'
            ).fillna(0)
        
        # Features pour segmentation
        features = ['age', 'income', 'credit_score', 'tx_count', 'tx_total', 'balance', 'recency_days']
        segmentation_features = segmentation_df[features].dropna()
        
        return segmentation_features

# Enregistrer dans le registre
from .base import DatasetRegistry
DatasetRegistry.register("financial", FinancialTransactionsLoader)
DatasetRegistry.register("financial-synthetic", lambda data_dir, **kwargs: 
                        FinancialTransactionsLoader(data_dir, dataset="synthetic", **kwargs))
DatasetRegistry.register("financial-kaggle", lambda data_dir, **kwargs: 
                        FinancialTransactionsLoader(data_dir, dataset="kaggle", **kwargs))
