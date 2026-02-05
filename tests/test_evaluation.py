"""
Tests pour le module d'évaluation MT5D
"""
import pytest
import numpy as np
import torch
import pandas as pd

from mt5d.evaluation.metrics import (
    MT5DMetrics, 
    compute_dimension_specific_metrics,
    BenchmarkMetrics
)

class TestMT5DMetrics:
    
    @pytest.fixture
    def metrics_calculator(self):
        return MT5DMetrics()
    
    @pytest.fixture
    def sample_classification_data(self):
        """Données de test pour la classification"""
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.rand(n_samples, 2)
        
        return y_pred, y_true
    
    @pytest.fixture
    def sample_regression_data(self):
        """Données de test pour la régression"""
        n_samples = 100
        y_true = np.random.randn(n_samples)
        y_pred = y_true + np.random.randn(n_samples) * 0.1
        
        return y_pred, y_true
    
    @pytest.fixture 
    def sample_clustering_data(self):
        """Données de test pour le clustering"""
        n_samples = 100
        embeddings = np.random.randn(n_samples, 10)
        clusters = np.random.randint(0, 5, n_samples)
        labels = np.random.randint(0, 5, n_samples)
        
        return embeddings, clusters, labels
    
    def test_compute_classification_metrics(self, metrics_calculator, 
                                          sample_classification_data):
        """Test les métriques de classification"""
        y_pred, y_true = sample_classification_data
        
        metrics = metrics_calculator._compute_classification_metrics(y_pred, y_true)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        if len(np.unique(y_true)) > 1:
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1
    
    def test_compute_regression_metrics(self, metrics_calculator,
                                      sample_regression_data):
        """Test les métriques de régression"""
        y_pred, y_true = sample_regression_data
        
        metrics = metrics_calculator._compute_regression_metrics(y_pred, y_true)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2_score'] <= 1  # R² peut être négatif
        
        # Vérifier que RMSE = sqrt(MSE)
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))
    
    def test_compute_clustering_metrics(self, metrics_calculator,
                                      sample_clustering_data):
        """Test les métriques de clustering"""
        embeddings, clusters, labels = sample_clustering_data
        
        metrics = metrics_calculator._compute_clustering_metrics(
            clusters, labels, None
        )
        
        assert 'n_clusters' in metrics
        assert metrics['n_clusters'] == len(np.unique(clusters))
        
        if len(np.unique(clusters)) > 1:
            assert 'adjusted_rand_score' in metrics
            assert 'normalized_mutual_info' in metrics
            assert -1 <= metrics['adjusted_rand_score'] <= 1
            assert 0 <= metrics['normalized_mutual_info'] <= 1
    
    def test_compute_rare_category_metrics(self, metrics_calculator):
        """Test les métriques pour catégories rares"""
        n_samples = 1000
        y_true = np.array([0] * 900 + [1] * 90 + [2] * 10)  # Catégories rares: 1 et 2
        y_pred = np.random.randint(0, 3, n_samples)
        
        metrics = metrics_calculator._compute_rare_category_metrics(y_true, y_pred)
        
        if metrics:  # Peut être vide si pas de catégories rares
            assert 'rare_category_precision_mean' in metrics
            assert 'rare_category_recall_mean' in metrics
            assert 'rare_category_f1_mean' in metrics
            assert 'n_rare_categories_found' in metrics
            
            assert 0 <= metrics['rare_category_precision_mean'] <= 1
            assert 0 <= metrics['rare_category_recall_mean'] <= 1
            assert 0 <= metrics['rare_category_f1_mean'] <= 1
            assert metrics['n_rare_categories_found'] >= 0
    
    def test_compute_all_metrics(self, metrics_calculator,
                               sample_classification_data,
                               sample_regression_data):
        """Test le calcul de toutes les métriques"""
        y_pred_cls, y_true_cls = sample_classification_data
        y_pred_reg, y_true_reg = sample_regression_data
        
        predictions = {
            'classification': y_pred_cls,
            'regression': y_pred_reg,
            'clustering': np.random.randint(0, 5, 100)
        }
        
        ground_truth = {
            'classification': y_true_cls,
            'regression': y_true_reg,
            'clustering': np.random.randint(0, 5, 100)
        }
        
        metrics = metrics_calculator.compute_all_metrics(
            predictions, ground_truth
        )
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Vérifier que les résultats sont enregistrés
        assert metrics_calculator.results == metrics
    
    def test_create_summary_report(self, metrics_calculator,
                                 sample_classification_data):
        """Test la création du rapport de métriques"""
        y_pred, y_true = sample_classification_data
        
        metrics = metrics_calculator._compute_classification_metrics(y_pred, y_true)
        metrics_calculator.results = metrics
        
        report = metrics_calculator.create_summary_report()
        
        assert isinstance(report, pd.DataFrame)
        assert not report.empty
        assert 'Category' in report.columns
        assert 'Metric' in report.columns
        assert 'Value' in report.columns
        assert 'Description' in report.columns

class TestDimensionSpecificMetrics:
    
    @pytest.fixture
    def sample_profiler_results(self):
        """Résultats de profilage de test"""
        return {
            'volume': {
                'total_rows': 10000,
                'compression_ratio': 0.5
            },
            'many_variables': {
                'total_columns': 50,
                'redundancy_score': 0.3
            },
            'high_cardinality': {
                'col1': {'is_high_cardinality': True, 'normalized_entropy': 0.8},
                'col2': {'is_high_cardinality': False, 'normalized_entropy': 0.2},
                'col3': {'is_high_cardinality': True, 'normalized_entropy': 0.9}
            },
            'many_tables': {
                'table_count': 5,
                'relationship_density': 0.6,
                'schema_complexity': 0.8
            },
            'repeated_measurements': {
                'table1': {'time_columns': ['date'], 'potential_id_columns': ['id']},
                'table2': {'time_columns': [], 'potential_id_columns': []},
                'table3': {'time_columns': ['timestamp'], 'potential_id_columns': ['entity_id']}
            }
        }
    
    def test_compute_dimension_specific_metrics(self, sample_profiler_results):
        """Test le calcul des métriques dimensionnelles"""
        metrics = compute_dimension_specific_metrics(sample_profiler_results)
        
        assert 'volume_score' in metrics
        assert 'memory_efficiency' in metrics
        assert 'variable_count' in metrics
        assert 'redundancy_score' in metrics
        assert 'high_cardinality_columns' in metrics
        assert 'table_count' in metrics
        assert 'relationship_density' in metrics
        assert 'schema_complexity' in metrics
        assert 'tables_with_repeated_measurements' in metrics
        assert 'longitudinality_score' in metrics
        
        # Vérifier les valeurs
        assert metrics['high_cardinality_columns'] == 2  # col1 et col3
        assert metrics['table_count'] == 5
        assert metrics['tables_with_repeated_measurements'] == 2  # table1 et table3
        assert np.isclose(metrics['longitudinality_score'], 2/3)  # 2 tables sur 3
    
    def test_empty_profiler_results(self):
        """Test avec résultats de profilage vides"""
        metrics = compute_dimension_specific_metrics({})
        
        # Doit retourner un dict vide ou avec valeurs par défaut
        assert isinstance(metrics, dict)

class TestBenchmarkMetrics:
    
    @pytest.fixture
    def benchmark_metrics(self):
        return BenchmarkMetrics()
    
    def test_compute_speed_metrics(self, benchmark_metrics):
        """Test les métriques de vitesse"""
        times = {
            'profiling': 1.5,
            'hypergraph': 3.2,
            'modeling': 10.1
        }
        
        sizes = {
            'profiling': 10000,
            'hypergraph': 5000,
            'modeling': 1000
        }
        
        metrics = benchmark_metrics.compute_speed_metrics(times, sizes)
        
        assert 'profiling_throughput' in metrics
        assert 'hypergraph_throughput' in metrics
        assert 'modeling_throughput' in metrics
        
        # Vérifier les calculs
        assert np.isclose(metrics['profiling_throughput'], 10000 / 1.5)
        assert np.isclose(metrics['hypergraph_throughput'], 5000 / 3.2)
        assert np.isclose(metrics['modeling_throughput'], 1000 / 10.1)
    
    def test_compute_memory_metrics(self, benchmark_metrics):
        """Test les métriques d'utilisation mémoire"""
        memory_usage = {
            'profiling': 500.0,  # MB
            'hypergraph': 1200.0,
            'modeling': 800.0
        }
        
        metrics = benchmark_metrics.compute_memory_metrics(memory_usage)
        
        assert 'peak_memory_mb' in metrics
        assert 'avg_memory_mb' in metrics
        assert 'memory_variation' in metrics
        
        assert metrics['peak_memory_mb'] == 1200.0
        assert np.isclose(metrics['avg_memory_mb'], np.mean([500, 1200, 800]))
        assert metrics['memory_variation'] >= 0
    
    def test_compute_scalability_metrics(self, benchmark_metrics):
        """Test les métriques de scalabilité"""
        metrics_per_size = {
            'size_1000': {
                'time': 1.0,
                'memory': 100.0,
                'accuracy': 0.85
            },
            'size_10000': {
                'time': 8.5,
                'memory': 800.0,
                'accuracy': 0.87
            },
            'size_100000': {
                'time': 75.0,
                'memory': 7500.0,
                'accuracy': 0.86
            }
        }
        
        scalability = benchmark_metrics.compute_scalability_metrics(metrics_per_size)
        
        assert 'time_avg_growth' in scalability
        assert 'memory_avg_growth' in scalability
        assert 'time_growth_std' in scalability
        assert 'memory_growth_std' in scalability
        
        # Vérifier les calculs de croissance
        time_growth_1 = 8.5 / 1.0
        time_growth_2 = 75.0 / 8.5
        expected_avg_growth = (time_growth_1 + time_growth_2) / 2
        
        assert np.isclose(scalability['time_avg_growth'], expected_avg_growth)

def test_metric_descriptions():
    """Test que toutes les métriques ont des descriptions"""
    calculator = MT5DMetrics()
    
    # Liste de métriques communes
    test_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'mse', 'rmse', 'mae', 'r2_score',
        'silhouette_score', 'adjusted_rand_score'
    ]
    
    for metric in test_metrics:
        description = calculator._get_metric_description(metric)
        assert description != 'No description available'
        assert len(description) > 0

def test_edge_cases():
    """Test les cas limites"""
    calculator = MT5DMetrics()
    
    # Test avec données vides
    empty_metrics = calculator._compute_classification_metrics(None, None)
    assert empty_metrics == {}
    
    # Test avec tenseurs PyTorch
    y_true_tensor = torch.tensor([0, 1, 0, 1])
    y_pred_tensor = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    
    metrics = calculator._compute_classification_metrics(
        y_pred_tensor, y_true_tensor
    )
    
    assert 'accuracy' in metrics
    assert isinstance(metrics['accuracy'], float)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
