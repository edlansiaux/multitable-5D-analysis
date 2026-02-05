"""
Configuration globale pour pytest
"""
import pytest
import pandas as pd
import numpy as np
import torch
import dgl

# Fixer les seeds pour la reproductibilité
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Fixer les seeds aléatoires pour tous les tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield

# Fixtures globales
@pytest.fixture
def device():
    """Retourne le device à utiliser (CPU ou GPU si disponible)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration de DGL
@pytest.fixture(scope='session', autouse=True)
def setup_dgl():
    """Configuration de DGL pour les tests"""
    dgl.seed(42)
    yield

# Désactiver le wandb pour les tests
@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch):
    """Mock wandb pour éviter les appels réseau pendant les tests"""
    monkeypatch.setenv('WANDB_MODE', 'disabled')

# Fixture pour les données temporelles
@pytest.fixture
def temporal_data():
    """Génère des données temporelles de test"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum()
    
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

# Fixture pour les données à haute cardinalité
@pytest.fixture
def high_cardinality_data():
    """Génère des données avec haute cardinalité"""
    n_samples = 1000
    n_categories = 500  # Haute cardinalité
    
    return pd.DataFrame({
        'id': range(n_samples),
        'category': np.random.choice([f'cat_{i}' for i in range(n_categories)], n_samples),
        'value': np.random.randn(n_samples)
    })
