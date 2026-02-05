"""
Métriques d'évaluation pour l'analyse multitable 5D
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)
import torch
import dgl

class MT5DMetrics:
    """Calculateur de métriques pour l'analyse 5D"""
    
    def __init__(self):
        self.results = {}
    
    def compute_all_metrics(self, predictions: Dict[str, Any], 
                          ground_truth: Dict[str, Any],
                          hypergraph: Optional[Any] = None) -> Dict[str, float]:
        """Calcule toutes les métriques pertinentes"""
        
        metrics = {}
        
        # 1. Métriques de classification
        if 'classification' in predictions:
            cls_metrics = self._compute_classification_metrics(
                predictions['classification'],
                ground_truth.get('classification')
            )
            metrics.update(cls_metrics)
        
        # 2. Métriques de régression
        if 'regression' in predictions:
            reg_metrics = self._compute_regression_metrics(
                predictions['regression'],
                ground_truth.get('regression')
            )
            metrics.update(reg_metrics)
        
        # 3. Métriques de clustering
        if 'clustering' in predictions:
            cluster_metrics = self._compute_clustering_metrics(
                predictions['clustering'],
                ground_truth.get('clustering'),
                hypergraph
            )
            metrics.update(cluster_metrics)
        
        # 4. Métriques relationnelles
        if hypergraph is not None:
            rel_metrics = self._compute_relational_metrics(hypergraph)
            metrics.update(rel_metrics)
        
        # 5. Métriques temporelles
        if 'temporal' in predictions:
            temp_metrics = self._compute_temporal_metrics(
                predictions['temporal'],
                ground_truth.get('temporal')
            )
            metrics.update(temp_metrics)
        
        self.results = metrics
        return metrics
    
    def _compute_classification_metrics(self, y_pred, y_true) -> Dict[str, float]:
        """Calcule les métriques de classification"""
        
        if y_true is None or y_pred is None:
            return {}
        
        # Convertir en numpy si besoin
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        
        metrics = {}
        
        try:
            # Accuracy
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred_labels = np.argmax(y_pred, axis=1)
                if y_true.ndim > 1 and y_true.shape[1] > 1:
                    y_true_labels = np.argmax(y_true, axis=1)
                else:
                    y_true_labels = y_true
            else:
                y_pred_labels = (y_pred > 0.5).astype(int)
                y_true_labels = y_true
            
            metrics['accuracy'] = accuracy_score(y_true_labels, y_pred_labels)
            
            # Precision, Recall, F1 (pour classification binaire/multiclasse)
            if len(np.unique(y_true_labels)) > 1:
                average = 'binary' if len(np.unique(y_true_labels)) == 2 else 'weighted'
                
                metrics['precision'] = precision_score(
                    y_true_labels, y_pred_labels, average=average, zero_division=0
                )
                metrics['recall'] = recall_score(
                    y_true_labels, y_pred_labels, average=average, zero_division=0
                )
                metrics['f1_score'] = f1_score(
                    y_true_labels, y_pred_labels, average=average, zero_division=0
                )
            
            # AUC-ROC (pour classification binaire)
            if len(np.unique(y_true)) == 2 and y_pred.ndim == 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            
            # Précision pour catégories rares
            if len(np.unique(y_true_labels)) > 2:
                rare_metrics = self._compute_rare_category_metrics(
                    y_true_labels, y_pred_labels
                )
                metrics.update(rare_metrics)
                
        except Exception as e:
            print(f"Erreur calcul métriques classification: {e}")
        
        return metrics
    
    def _compute_regression_metrics(self, y_pred, y_true) -> Dict[str, float]:
        """Calcule les métriques de régression"""
        
        if y_true is None or y_pred is None:
            return {}
        
        # Convertir en numpy si besoin
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        
        metrics = {}
        
        try:
            # MSE, RMSE, MAE
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2_score'] = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Explained variance
            metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
            
        except Exception as e:
            print(f"Erreur calcul métriques régression: {e}")
        
        return metrics
    
    def _compute_clustering_metrics(self, clusters, labels=None, hypergraph=None) -> Dict[str, float]:
        """Calcule les métriques de clustering"""
        
        metrics = {}
        
        try:
            if clusters is None:
                return metrics
            
            # Convertir en numpy si besoin
            if torch.is_tensor(clusters):
                clusters = clusters.cpu().numpy()
            
            # Silhouette score (nécessite des features)
            if hypergraph is not None and hasattr(hypergraph, 'nodes'):
                node_features = hypergraph.nodes['node'].data.get('feat', None)
                if node_features is not None:
                    if torch.is_tensor(node_features):
                        node_features = node_features.cpu().numpy()
                    
                    if len(np.unique(clusters)) > 1:
                        metrics['silhouette_score'] = silhouette_score(
                            node_features, clusters
                        )
            
            # Métriques basées sur les labels vrais (si disponibles)
            if labels is not None:
                if torch.is_tensor(labels):
                    labels = labels.cpu().numpy()
                
                metrics['adjusted_rand_score'] = adjusted_rand_score(labels, clusters)
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                    labels, clusters
                )
            
            # Nombre de clusters et distribution
            unique_clusters = np.unique(clusters)
            metrics['n_clusters'] = len(unique_clusters)
            
            # Équilibre des clusters
            cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
            if cluster_sizes:
                cluster_sizes = np.array(cluster_sizes)
                metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)
            
        except Exception as e:
            print(f"Erreur calcul métriques clustering: {e}")
        
        return metrics
    
    def _compute_relational_metrics(self, hypergraph) -> Dict[str, float]:
        """Calcule les métriques relationnelles"""
        
        metrics = {}
        
        try:
            if hypergraph is None:
                return metrics
            
            # Statistiques basiques du graphe
            metrics['num_nodes'] = hypergraph.num_nodes()
            metrics['num_edges'] = hypergraph.num_edges()
            
            if hypergraph.num_edges() > 0:
                metrics['edge_density'] = (2 * hypergraph.num_edges()) / \
                                         (hypergraph.num_nodes() * (hypergraph.num_nodes() - 1))
            
            # Centralité (approximation)
            if hasattr(hypergraph, 'in_degrees'):
                in_degrees = hypergraph.in_degrees()
                if torch.is_tensor(in_degrees):
                    in_degrees = in_degrees.cpu().numpy()
                
                metrics['avg_in_degree'] = np.mean(in_degrees)
                metrics['max_in_degree'] = np.max(in_degrees)
                metrics['degree_std'] = np.std(in_degrees)
            
            # Métriques pour hypergraphes
            if hasattr(hypergraph, 'etypes'):
                edge_types = hypergraph.edata.get('type', None)
                if edge_types is not None:
                    if torch.is_tensor(edge_types):
                        edge_types = edge_types.cpu().numpy()
                    
                    unique_types = np.unique(edge_types)
                    metrics['num_edge_types'] = len(unique_types)
                    
                    # Distribution des types d'arêtes
                    type_counts = np.bincount(edge_types)
                    if len(type_counts) > 0:
                        metrics['edge_type_balance'] = np.std(type_counts) / np.mean(type_counts)
            
        except Exception as e:
            print(f"Erreur calcul métriques relationnelles: {e}")
        
        return metrics
    
    def _compute_temporal_metrics(self, predictions, ground_truth) -> Dict[str, float]:
        """Calcule les métriques temporelles"""
        
        metrics = {}
        
        try:
            if predictions is None or ground_truth is None:
                return metrics
            
            # Convertir en numpy si besoin
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.cpu().numpy()
            
            # Pour séries temporelles
            if predictions.ndim == 2 and ground_truth.ndim == 2:
                # MSE par pas de temps
                mse_per_step = np.mean((predictions - ground_truth) ** 2, axis=0)
                metrics['mse_temporal_mean'] = np.mean(mse_per_step)
                metrics['mse_temporal_std'] = np.std(mse_per_step)
                
                # MAE par pas de temps
                mae_per_step = np.mean(np.abs(predictions - ground_truth), axis=0)
                metrics['mae_temporal_mean'] = np.mean(mae_per_step)
                metrics['mae_temporal_std'] = np.std(mae_per_step)
            
            # Pour prédiction d'événements
            elif predictions.ndim == 1 and ground_truth.ndim == 1:
                # Erreur de timing
                time_error = np.abs(predictions - ground_truth)
                metrics['mean_time_error'] = np.mean(time_error)
                metrics['std_time_error'] = np.std(time_error)
                
                # Précision de détection d'événements
                threshold = np.percentile(ground_truth, 90)  # Exemple
                pred_events = predictions > threshold
                true_events = ground_truth > threshold
                
                if np.any(true_events):
                    metrics['event_precision'] = np.sum(pred_events & true_events) / \
                                               (np.sum(pred_events) + 1e-10)
                    metrics['event_recall'] = np.sum(pred_events & true_events) / \
                                            (np.sum(true_events) + 1e-10)
            
        except Exception as e:
            print(f"Erreur calcul métriques temporelles: {e}")
        
        return metrics
    
    def _compute_rare_category_metrics(self, y_true, y_pred, rare_threshold: float = 0.01) -> Dict[str, float]:
        """Calcule les métriques pour catégories rares"""
        
        metrics = {}
        
        try:
            # Identifier les catégories rares
            unique, counts = np.unique(y_true, return_counts=True)
            total = len(y_true)
            
            rare_categories = unique[counts / total < rare_threshold]
            
            if len(rare_categories) == 0:
                return metrics
            
            # Précision et rappel pour catégories rares
            rare_precision = []
            rare_recall = []
            
            for cat in rare_categories:
                # Précision pour cette catégorie
                true_pos = np.sum((y_pred == cat) & (y_true == cat))
                pred_pos = np.sum(y_pred == cat)
                
                precision = true_pos / (pred_pos + 1e-10)
                rare_precision.append(precision)
                
                # Rappel pour cette catégorie
                actual_pos = np.sum(y_true == cat)
                recall = true_pos / (actual_pos + 1e-10)
                rare_recall.append(recall)
            
            metrics['rare_category_precision_mean'] = np.mean(rare_precision)
            metrics['rare_category_recall_mean'] = np.mean(rare_recall)
            metrics['rare_category_f1_mean'] = 2 * (
                metrics['rare_category_precision_mean'] * metrics['rare_category_recall_mean']
            ) / (metrics['rare_category_precision_mean'] + metrics['rare_category_recall_mean'] + 1e-10)
            
            # Nombre de catégories rares correctement prédites
            metrics['n_rare_categories_found'] = len([
                cat for cat in rare_categories 
                if np.any(y_pred[y_true == cat] == cat)
            ])
            
        except Exception as e:
            print(f"Erreur calcul métriques catégories rares: {e}")
        
        return metrics
    
    def create_summary_report(self) -> pd.DataFrame:
        """Crée un rapport détaillé des métriques"""
        
        if not self.results:
            return pd.DataFrame()
        
        # Catégoriser les métriques
        categories = {
            'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
            'regression': ['mse', 'rmse', 'mae', 'r2_score', 'explained_variance'],
            'clustering': ['silhouette_score', 'adjusted_rand_score', 
                          'normalized_mutual_info', 'n_clusters', 'cluster_balance'],
            'relational': ['num_nodes', 'num_edges', 'edge_density', 
                          'avg_in_degree', 'max_in_degree', 'degree_std'],
            'temporal': ['mse_temporal_mean', 'mse_temporal_std', 
                        'mae_temporal_mean', 'mae_temporal_std'],
            'rare_categories': ['rare_category_precision_mean', 
                              'rare_category_recall_mean', 'rare_category_f1_mean',
                              'n_rare_categories_found']
        }
        
        # Créer le DataFrame
        report_data = []
        for category, metric_list in categories.items():
            for metric in metric_list:
                if metric in self.results:
                    report_data.append({
                        'Category': category,
                        'Metric': metric,
                        'Value': self.results[metric],
                        'Description': self._get_metric_description(metric)
                    })
        
        # Ajouter les métriques non catégorisées
        for metric, value in self.results.items():
            if not any(metric in ml for ml in categories.values()):
                report_data.append({
                    'Category': 'other',
                    'Metric': metric,
                    'Value': value,
                    'Description': self._get_metric_description(metric)
                })
        
        df = pd.DataFrame(report_data)
        return df.sort_values(['Category', 'Metric'])
    
    def _get_metric_description(self, metric: str) -> str:
        """Retourne la description d'une métrique"""
        
        descriptions = {
            'accuracy': 'Proportion de prédictions correctes',
            'precision': 'Précision des prédictions positives',
            'recall': 'Couverture des vrais positifs',
            'f1_score': 'Moyenne harmonique de précision et rappel',
            'roc_auc': 'Aire sous la courbe ROC',
            'mse': 'Erreur quadratique moyenne',
            'rmse': 'Racine de l\'erreur quadratique moyenne',
            'mae': 'Erreur absolue moyenne',
            'r2_score': 'Coefficient de détermination',
            'explained_variance': 'Variance expliquée par le modèle',
            'silhouette_score': 'Mesure de cohérence des clusters (-1 à 1)',
            'adjusted_rand_score': 'Similarité entre clusters et labels vrais',
            'normalized_mutual_info': 'Information mutuelle normalisée',
            'n_clusters': 'Nombre de clusters détectés',
            'cluster_balance': 'Écart-type relatif de la taille des clusters',
            'num_nodes': 'Nombre de nœuds dans le graphe',
            'num_edges': 'Nombre d\'arêtes dans le graphe',
            'edge_density': 'Densité du graphe',
            'avg_in_degree': 'Degré entrant moyen',
            'max_in_degree': 'Degré entrant maximum',
            'degree_std': 'Écart-type des degrés',
            'mse_temporal_mean': 'MSE moyen sur la dimension temporelle',
            'mse_temporal_std': 'Écart-type du MSE temporel',
            'mae_temporal_mean': 'MAE moyen sur la dimension temporelle',
            'mae_temporal_std': 'Écart-type du MAE temporel',
            'rare_category_precision_mean': 'Précision moyenne pour catégories rares',
            'rare_category_recall_mean': 'Rappel moyen pour catégories rares',
            'rare_category_f1_mean': 'F1 moyen pour catégories rares',
            'n_rare_categories_found': 'Nombre de catégories rares détectées'
        }
        
        return descriptions.get(metric, 'No description available')

def compute_dimension_specific_metrics(profiler_results: Dict[str, Any]) -> Dict[str, float]:
    """Calcule des métriques spécifiques aux 5 dimensions"""
    
    metrics = {}
    
    try:
        # Dimension 1: Volume
        if 'volume' in profiler_results:
            vol = profiler_results['volume']
            metrics['volume_score'] = min(10, vol.get('total_rows', 0) / 1e6)  # Normalisé
            metrics['memory_efficiency'] = vol.get('compression_ratio', 0)
        
        # Dimension 2: Many Variables
        if 'many_variables' in profiler_results:
            vars_ = profiler_results['many_variables']
            metrics['variable_count'] = vars_.get('total_columns', 0)
            metrics['redundancy_score'] = vars_.get('redundancy_score', 0)
        
        # Dimension 3: High Cardinality
        if 'high_cardinality' in profiler_results:
            high_card = profiler_results['high_cardinality']
            
            # Compter les colonnes à haute cardinalité
            high_card_cols = sum(
                1 for col_metrics in high_card.values() 
                if col_metrics.get('is_high_cardinality', False)
            )
            
            metrics['high_cardinality_columns'] = high_card_cols
            
            # Score moyen d'entropie
            entropies = [
                col_metrics.get('normalized_entropy', 0) 
                for col_metrics in high_card.values()
            ]
            if entropies:
                metrics['avg_entropy'] = np.mean(entropies)
        
        # Dimension 4: Many Tables
        if 'many_tables' in profiler_results:
            tables = profiler_results['many_tables']
            metrics['table_count'] = tables.get('table_count', 0)
            metrics['relationship_density'] = tables.get('relationship_density', 0)
            metrics['schema_complexity'] = tables.get('schema_complexity', 0)
        
        # Dimension 5: Repeated Measurements
        if 'repeated_measurements' in profiler_results:
            repeated = profiler_results['repeated_measurements']
            
            # Compter les tables avec mesures répétées
            tables_with_repeated = sum(
                1 for table_metrics in repeated.values() 
                if table_metrics.get('time_columns') or table_metrics.get('potential_id_columns')
            )
            
            metrics['tables_with_repeated_measurements'] = tables_with_repeated
            
            # Score de longitudinalité
            total_tables = len(repeated)
            metrics['longitudinality_score'] = tables_with_repeated / total_tables if total_tables > 0 else 0
        
    except Exception as e:
        print(f"Erreur calcul métriques dimensionnelles: {e}")
    
    return metrics

class BenchmarkMetrics:
    """Métriques pour le benchmarking"""
    
    @staticmethod
    def compute_speed_metrics(times: Dict[str, float], 
                            sizes: Dict[str, int]) -> Dict[str, float]:
        """Calcule les métriques de vitesse"""
        
        metrics = {}
        
        for stage, time in times.items():
            if stage in sizes:
                # Items par seconde
                metrics[f'{stage}_throughput'] = sizes[stage] / time if time > 0 else 0
            
            # Temps normalisé
            if 'small' in times and stage in times:
                metrics[f'{stage}_speedup'] = times['small'] / time if time > 0 else 0
        
        return metrics
    
    @staticmethod
    def compute_memory_metrics(memory_usage: Dict[str, float]) -> Dict[str, float]:
        """Calcule les métriques d'utilisation mémoire"""
        
        metrics = {}
        
        if memory_usage:
            metrics['peak_memory_mb'] = max(memory_usage.values())
            metrics['avg_memory_mb'] = np.mean(list(memory_usage.values()))
            metrics['memory_variation'] = np.std(list(memory_usage.values()))
        
        return metrics
    
    @staticmethod
    def compute_scalability_metrics(metrics_per_size: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calcule les métriques de scalabilité"""
        
        scalability = {}
        
        if len(metrics_per_size) < 2:
            return scalability
        
        sizes = sorted(metrics_per_size.keys(), 
                      key=lambda x: int(x.replace('size_', '')) if 'size_' in x else 0)
        
        for metric in list(metrics_per_size[sizes[0]].keys()):
            values = [metrics_per_size[size].get(metric, 0) for size in sizes]
            
            if len(values) > 1 and values[0] > 0:
                # Taux de croissance
                growth_rates = []
                for i in range(1, len(values)):
                    if values[i-1] > 0:
                        growth_rates.append(values[i] / values[i-1])
                
                if growth_rates:
                    scalability[f'{metric}_avg_growth'] = np.mean(growth_rates)
                    scalability[f'{metric}_growth_std'] = np.std(growth_rates)
        
        return scalability
