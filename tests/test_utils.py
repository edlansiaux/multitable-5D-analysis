"""
Tests pour les utilitaires MT5D
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mt5d.utils.io import DataLoader, DataValidator, create_sample_dataset
from mt5d.utils.visualization import MT5DVisualizer, plot_correlation_matrix

class TestDataLoader:
    
    @pytest.fixture
    def temp_dir(self):
        """Crée un répertoire temporaire pour les tests"""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def sample_data(self):
        """Données de test"""
        return {
            'table1': pd.DataFrame({
                'id': [1, 2, 3],
                'value': [10.0, 20.0, 30.0]
            }),
            'table2': pd.DataFrame({
                'id': [1, 2, 3],
                'category': ['A', 'B', 'C']
            })
        }
    
    def test_create_sample_dataset(self, temp_dir):
        """Test la création d'un dataset d'exemple"""
        create_sample_dataset(temp_dir, n_samples=100)
        
        # Vérifier que les fichiers sont créés
        assert (temp_dir / 'patients.parquet').exists()
        assert (temp_dir / 'diagnoses.parquet').exists()
        assert (temp_dir / 'visits.parquet').exists()
        assert (temp_dir / 'medications.parquet').exists()
        assert (temp_dir / 'relationships.json').exists()
    
    def test_load_from_directory(self, temp_dir, sample_data):
        """Test le chargement depuis un répertoire"""
        # Sauvegarder les données
        for name, df in sample_data.items():
            df.to_parquet(temp_dir / f'{name}.parquet')
        
        # Charger
        loader = DataLoader()
        loaded = loader.load_from_directory(temp_dir, format='parquet')
        
        assert len(loaded) == 2
        assert 'table1' in loaded
        assert 'table2' in loaded
        assert len(loaded['table1']) == 3
        assert len(loaded['table2']) == 3
    
    def test_load_relationships(self, temp_dir):
        """Test le chargement des relations"""
        # Créer un fichier de relations
        relationships = [
            {
                'source_table': 'table1',
                'source_column': 'id',
                'target_table': 'table2',
                'target_column': 'id',
                'relation_type': 'related_to'
            }
        ]
        
        import json
        with open(temp_dir / 'relationships.json', 'w') as f:
            json.dump(relationships, f)
        
        # Charger
        loader = DataLoader()
        loaded = loader.load_relationships(temp_dir / 'relationships.json')
        
        assert len(loaded) == 1
        assert loaded[0] == ('table1', 'id', 'table2', 'id', 'related_to')
    
    def test_save_and_load_results(self, temp_dir, sample_data):
        """Test la sauvegarde et chargement des résultats"""
        loader = DataLoader()
        
        # Résultats de test
        results = {
            'tables': sample_data,
            'metrics': {'accuracy': 0.95},
            'list_data': [1, 2, 3],
            'dict_data': {'key': 'value'}
        }
        
        # Sauvegarder
        save_dir = temp_dir / 'saved_results'
        loader.save_results(results, save_dir)
        
        # Vérifier que les fichiers sont créés
        assert (save_dir / 'tables').exists()
        assert (save_dir / 'metrics.json').exists()
        assert (save_dir / 'list_data.json').exists()
        assert (save_dir / 'dict_data.json').exists()
        
        # Charger
        loaded = loader.load_results(save_dir)
        
        assert 'tables' in loaded
        assert 'metrics' in loaded
        assert loaded['metrics']['accuracy'] == 0.95

class TestDataValidator:
    
    @pytest.fixture
    def sample_tables(self):
        """Tables de test pour la validation"""
        return {
            'patients': pd.DataFrame({
                'patient_id': [1, 2, 3, 4, 5],
                'age': [25, 30, 35, None, 45],
                'gender': ['M', 'F', 'M', 'F', 'M']
            }),
            'visits': pd.DataFrame({
                'visit_id': [1, 2, 3],
                'patient_id': [1, 2, 99],  # 99 n'existe pas dans patients
                'date': pd.date_range('2023-01-01', periods=3)
            })
        }
    
    @pytest.fixture
    def sample_relationships(self):
        """Relations de test"""
        return [
            ('visits', 'patient_id', 'patients', 'patient_id', 'patient_visit')
        ]
    
    def test_validate_tables(self, sample_tables):
        """Test la validation des tables"""
        validator = DataValidator()
        
        is_valid = validator.validate(sample_tables)
        
        # Doit avoir des avertissements pour valeurs manquantes
        assert len(validator.warnings) > 0
        assert any('missing' in str(w).lower() for w in validator.warnings)
    
    def test_validate_relationships(self, sample_tables, sample_relationships):
        """Test la validation des relations"""
        validator = DataValidator()
        
        is_valid = validator.validate(sample_tables, sample_relationships)
        
        # Doit avoir une erreur pour la relation invalide (patient_id 99)
        assert len(validator.errors) > 0
    
    def test_validate_data_quality(self, sample_tables):
        """Test la validation de la qualité des données"""
        validator = DataValidator()
        
        # Ajouter une colonne numérique avec outliers
        sample_tables['patients']['outlier_col'] = [1, 2, 3, 4, 1000]
        
        is_valid = validator.validate(sample_tables)
        
        # Doit détecter les outliers
        assert any('outlier' in str(w).lower() for w in validator.warnings)

class TestMT5DVisualizer:
    
    @pytest.fixture
    def visualizer(self):
        return MT5DVisualizer()
    
    @pytest.fixture
    def sample_metrics(self):
        """Métriques de test pour le radar chart"""
        return {
            'volume': {'score': 8.5},
            'many_variables': {'score': 7.2},
            'high_cardinality': {'score': 9.0},
            'many_tables': {'score': 8.8},
            'repeated_measurements': {'score': 6.5}
        }
    
    def test_plot_dimension_radar(self, visualizer, sample_metrics):
        """Test la création du radar chart"""
        fig = visualizer.plot_dimension_radar(sample_metrics)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_temporal_patterns(self, visualizer):
        """Test la visualisation des patterns temporels"""
        # Créer des données temporelles de test
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(100).cumsum(),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'entity_id': np.random.choice(['ent1', 'ent2', 'ent3'], 100)
        })
        
        fig = visualizer.plot_temporal_patterns(data)
        
        assert fig is not None
        assert len(fig.axes) == 4  # 4 sous-graphiques
    
    def test_plot_embedding_space(self, visualizer):
        """Test la visualisation de l'espace d'embedding"""
        # Créer des embeddings de test
        embeddings = np.random.randn(100, 50)
        labels = np.random.randint(0, 3, 100)
        
        fig = visualizer.plot_embedding_space(embeddings, labels, method='pca')
        
        assert fig is not None

def test_plot_correlation_matrix():
    """Test la création de la matrice de corrélations"""
    # Créer des tables de test avec colonnes numériques
    tables = {
        'table1': pd.DataFrame({
            'col1': np.random.randn(100),
            'col2': np.random.randn(100)
        }),
        'table2': pd.DataFrame({
            'col3': np.random.randn(100),
            'col4': np.random.randn(100)
        })
    }
    
    fig, corr_matrix = plot_correlation_matrix(tables)
    
    assert fig is not None
    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (4, 4)  # 4 colonnes totales
    
    # Vérifier que la matrice est symétrique
    assert np.allclose(corr_matrix.values, corr_matrix.values.T)

def test_data_validator_edge_cases():
    """Test les cas limites du validateur"""
    validator = DataValidator()
    
    # Test avec tables vides
    empty_tables = {
        'empty': pd.DataFrame(),
        'with_data': pd.DataFrame({'col': [1, 2, 3]})
    }
    
    is_valid = validator.validate(empty_tables)
    assert not is_valid  # Doit échouer car table vide
    
    # Test avec doublons de noms de colonnes
    df_with_duplicates = pd.DataFrame({
        'col': [1, 2, 3],
        'col': [4, 5, 6]  # Duplicate column name
    })
    
    tables_with_duplicates = {'test': df_with_duplicates}
    is_valid = validator.validate(tables_with_duplicates)
    assert not is_valid  # Doit échouer car colonnes en double

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
