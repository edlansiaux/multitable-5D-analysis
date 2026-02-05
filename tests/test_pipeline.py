import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from mt5d.core.pipeline import MT5DPipeline

class TestMT5DPipeline:
    
    @pytest.fixture
    def sample_data(self):
        """Données de test pour le pipeline"""
        patients = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [30, 40, 50],
            'gender': ['M', 'F', 'M']
        })
        
        labs = pd.DataFrame({
            'lab_id': [101, 102, 103, 104],
            'patient_id': [1, 1, 2, 3],
            'value': [4.5, 4.8, 3.9, 5.2],
            'timestamp': pd.to_datetime(['2020-01-01', '2020-01-02', 
                                       '2020-01-03', '2020-01-04'])
        })
        
        return {
            'patients': patients,
            'labs': labs
        }
    
    @pytest.fixture
    def sample_relationships(self):
        return [
            ('labs', 'patient_id', 'patients', 'patient_id', 'has_lab')
        ]
    
    @pytest.fixture
    def temp_dir(self):
        """Répertoire temporaire pour les tests"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_pipeline_initialization(self):
        """Test l'initialisation du pipeline"""
        # Avec config par défaut
        pipeline = MT5DPipeline()
        assert pipeline.config == {}
        assert pipeline.profiler is not None
        assert pipeline.builder is not None
        
        # Avec fichier de config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            f.write("pipeline:\n  name: 'test'\n")
            f.flush()
            
            pipeline = MT5DPipeline(f.name)
            assert pipeline.config['pipeline']['name'] == 'test'
    
    def test_pipeline_run_complete(self, sample_data, sample_relationships):
        """Test l'exécution complète du pipeline"""
        pipeline = MT5DPipeline()
        
        results = pipeline.run(
            tables=sample_data,
            relationships=sample_relationships,
            target_task="patient_classification"
        )
        
        # Vérifie la structure des résultats
        assert 'metrics' in results
        assert 'recommendations' in results
        assert 'hypergraph' in results
        assert 'features' in results
        assert 'results' in results
        assert 'evaluation' in results
        
        # Vérifie que le modèle est créé
        assert pipeline.model is not None
    
    def test_pipeline_run_without_target(self, sample_data, sample_relationships):
        """Test l'exécution sans tâche cible (insights généraux)"""
        pipeline = MT5DPipeline()
        
        results = pipeline.run(
            tables=sample_data,
            relationships=sample_relationships
        )
        
        assert 'results' in results
        # Doit générer des insights
        assert 'insights' in results['results'] or 'patterns' in results['results']
    
    def test_pipeline_save_load(self, sample_data, sample_relationships, temp_dir):
        """Test la sauvegarde et chargement du pipeline"""
        pipeline = MT5DPipeline()
        
        # Exécute le pipeline
        pipeline.run(sample_data, sample_relationships)
        
        # Sauvegarde
        save_path = Path(temp_dir) / "saved_pipeline"
        pipeline.save_pipeline(save_path)
        
        # Vérifie que les fichiers sont créés
        assert (save_path / 'config.yaml').exists()
        assert (save_path / 'results.pkl').exists()
        
        # Pour le modèle, il peut ne pas être sauvegardé s'il n'est pas entraîné
        # Ce test vérifie juste que la sauvegarde ne plante pas
    
    def test_pipeline_error_handling(self):
        """Test la gestion des erreurs"""
        pipeline = MT5DPipeline()
        
        # Données invalides
        invalid_data = {'empty': pd.DataFrame()}
        
        with pytest.raises(Exception):
            pipeline.run(invalid_data)
    
    def test_pipeline_large_dataset(self, temp_dir):
        """Test avec un dataset plus volumineux"""
        # Crée un dataset plus grand mais gérable
        n_patients = 100
        n_labs = 500
        
        patients = pd.DataFrame({
            'patient_id': range(n_patients),
            'age': np.random.randint(18, 90, n_patients),
            'gender': np.random.choice(['M', 'F'], n_patients)
        })
        
        labs = pd.DataFrame({
            'lab_id': range(n_labs),
            'patient_id': np.random.choice(range(n_patients), n_labs),
            'value': np.random.normal(0, 1, n_labs),
            'timestamp': pd.date_range('2020-01-01', periods=n_labs, freq='H')
        })
        
        data = {'patients': patients, 'labs': labs}
        relationships = [('labs', 'patient_id', 'patients', 'patient_id', 'has_lab')]
        
        pipeline = MT5DPipeline()
        
        # Doit s'exécuter sans erreur
        results = pipeline.run(data, relationships)
        
        assert results['metrics'].volume['total_rows'] == n_patients + n_labs
        assert results['hypergraph'] is not None
    
    def test_pipeline_config_overrides(self, sample_data):
        """Test la possibilité de surcharger la config"""
        config = {
            'profiling': {
                'high_cardinality_threshold': 500
            },
            'models': {
                'rht': {
                    'hidden_dim': 512
                }
            }
        }
        
        pipeline = MT5DPipeline()
        pipeline.config = config
        
        # Vérifie que la config est appliquée
        assert pipeline.profiler.config['high_cardinality_threshold'] == 500
