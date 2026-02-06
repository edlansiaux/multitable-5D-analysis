import pandas as pd
from typing import Dict, Tuple, List
from .base import BaseDataset

class FinancialDataset(BaseDataset):
    """
    Dataset Transactions Financières (Section 7.1.3).
    Tables : Accounts, Transactions, Merchants, Devices.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        
    def load(self) -> Tuple[Dict[str, pd.DataFrame], List[Tuple]]:
        print(f"Chargement du dataset Finance depuis {self.root_dir}...")
        
        tables = {
            'accounts': pd.DataFrame(columns=['account_id', 'owner_name', 'balance']),
            'merchants': pd.DataFrame(columns=['merchant_id', 'category_code']),
            'transactions': pd.DataFrame(columns=['trans_id', 'account_id', 'merchant_id', 'amount', 'timestamp']),
            'devices': pd.DataFrame(columns=['device_id', 'account_id', 'last_login'])
        }
        
        relationships = [
            ('accounts', 'account_id', 'transactions', 'account_id', 'one_to_many'),
            ('merchants', 'merchant_id', 'transactions', 'merchant_id', 'one_to_many'),
            ('accounts', 'account_id', 'devices', 'account_id', 'one_to_many')
        ]
        
        return tables, relationships

    def get_temporal_columns(self) -> Dict[str, str]:
        return {
            'transactions': 'timestamp',
            'devices': 'last_login'
        }
