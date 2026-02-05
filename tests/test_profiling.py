import pytest
import pandas as pd
import numpy as np
from mt5d.core.profiling import DimensionalProfiler, DimensionMetrics

class TestDimensionalProfiler:
    
    @pytest.fixture
    def sample_tables(self):
        """Crée des tables de test"""
        patients = pd.DataFrame({
            'patient_id': range(100),
            'age': np.random.randint(18, 90, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'diagnosis': np.random.choice(['I10', 'K37', 'M61.2', 'I41'], 100)
        })
        
        labs = pd.DataFrame({
            'lab_id': range(500),
            'patient_id': np.random.choice(range(100), 500),
            'parameter': np.random.choice(['K+', 'Na+', 'Glucose'], 500),
            'value': np.random.normal(0, 1, 500),
            'timestamp': pd.date_range('2020-01-01', periods=500, freq='H')
        })
        
        return {
            'patients': patients,
            'labs': labs
        }
    
    @pytest.fixture
    def sample_relationships(self):
        """Relations de test"""
        return [
            ('labs', 'patient_id', 'patients', 'patient_id', 'has_lab')
        ]
    
    def test_profiler_initialization(self):
        """Test l'initialisation du profiler"""
        profiler = DimensionalProfiler()
        assert profiler is not None
        assert profiler.config == {}
    
    def test_profile_volume(self, sample_tables):
        """Test la dimension Volume"""
        profiler = DimensionalProfiler()
        metrics = profiler.profile(sample_tables)
        
        assert 'total_rows' in metrics.volume
        assert metrics.volume['total_rows'] == 600  # 100 + 500
        
        assert metrics.volume['total_rows'] > 0
        assert metrics.volume['total_memory_mb'] > 0
    
    def test_profile_many_variables(self, sample_tables):
        """Test la dimension Many Variables"""
        profiler = DimensionalProfiler()
        metrics = profiler.profile(sample_tables)
        
        assert 'total_columns' in metrics.many_variables
        # patients(4) + labs(5) = 9 colonnes totales
        assert metrics.many_variables['total_columns'] == 9
    
    def test_profile_high_cardinality(self, sample_tables):
        """Test la dimension High Cardinality"""
        profiler = DimensionalProfiler()
        metrics = profiler.profile(sample_tables)
        
        # Vérifie qu'on détecte les colonnes catégorielles
        assert len(metrics.high_cardinality) > 0
        
        for col_metrics in metrics.high_cardinality.values():
            assert 'unique_count' in col_metrics
            assert 'entropy' in col_metrics
    
    def test_profile_many_tables(self, sample_tables, sample_relationships):
        """Test la dimension Many Tables"""
        profiler = DimensionalProfiler()
        metrics = profiler.profile(sample_tables, sample_relationships)
        
        assert metrics.many_tables['table_count'] == 2
        assert metrics.many_tables['relationship_count'] == 1
        assert 0 <= metrics.many_tables['relationship_density'] <= 1
    
    def test_profile_repeated_measurements(self, sample_tables):
        """Test la dimension Repeated Measurements"""
        profiler = DimensionalProfiler()
        metrics = profiler.profile(sample_tables)
        
        # Doit détecter la table labs comme ayant des mesures répétées
        assert 'labs' in metrics.repeated_measurements
        assert 'time_columns' in metrics.repeated_measurements['labs']
    
    def test_recommend_pipeline(self, sample_tables):
        """Test les recommandations du pipeline"""
        profiler = DimensionalProfiler()
        profiler.profile(sample_tables)
        recommendations = profiler.recommend_pipeline()
        
        assert 'compression_strategy' in recommendations
        assert 'model_architecture' in recommendations
        assert 'embedding_strategy' in recommendations
    
    def test_empty_tables(self):
        """Test avec tables vides"""
        profiler = DimensionalProfiler()
        empty_tables = {'empty': pd.DataFrame()}
        
        metrics = profiler.profile(empty_tables)
        assert metrics.volume['total_rows'] == 0
    
    def test_large_dataset(self):
        """Test avec dataset volumineux"""
        # Crée un dataset plus grand
        large_table = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice([f'cat_{i}' for i in range(100)], 10000)
        })
        
        profiler = DimensionalProfiler()
        metrics = profiler.profile({'large': large_table})
        
        assert metrics.volume['total_rows'] == 10000
        assert metrics.high_cardinality['large.category']['unique_count'] == 100
