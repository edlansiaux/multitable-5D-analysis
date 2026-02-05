"""
Exemple d'utilisation avancée du framework MT5D
"""
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from mt5d import MT5DPipeline
from mt5d.core.profiling import DimensionalProfiler
from mt5d.utils.io import DataLoader, DataValidator, create_sample_dataset
from mt5d.utils.visualization import MT5DVisualizer, plot_correlation_matrix
from mt5d.evaluation.metrics import MT5DMetrics, compute_dimension_specific_metrics
from mt5d.models.embeddings import PentE
from mt5d.models.architectures import RelationalHypergraphTransformer

def advanced_analysis_example():
    """Exemple complet d'analyse avancée"""
    
    print("=" * 80)
    print("MT5D FRAMEWORK - UTILISATION AVANCÉE")
    print("=" * 80)
    
    # 1. Création d'un dataset synthétique complexe
    print("\n1. Création d'un dataset synthétique complexe...")
    
    output_dir = Path("advanced_example_data")
    if not output_dir.exists():
        create_sample_dataset(output_dir, n_samples=5000)
    
    # 2. Chargement des données avec validation
    print("\n2. Chargement et validation des données...")
    
    loader = DataLoader()
    validator = DataValidator()
    
    # Charger les tables
    tables = loader.load_from_directory(output_dir, format='parquet')
    
    # Charger les relations
    relationships = loader.load_relationships(output_dir / 'relationships.json')
    
    # Valider les données
    is_valid = validator.validate(tables, relationships)
    
    if not is_valid:
        print("⚠ Données invalides, mais poursuite de l'analyse...")
    
    # 3. Profilage 5D approfondi
    print("\n3. Profilage 5D approfondi...")
    
    profiler = DimensionalProfiler(config={
        'detailed_analysis': True,
        'compute_correlations': True,
        'detect_temporal_patterns': True
    })
    
    metrics = profiler.profile(tables, relationships)
    
    # Calculer des métriques dimensionnelles avancées
    dimension_metrics = compute_dimension_specific_metrics({
        'volume': metrics.volume,
        'many_variables': metrics.many_variables,
        'high_cardinality': metrics.high_cardinality,
        'many_tables': metrics.many_tables,
        'repeated_measurements': metrics.repeated_measurements
    })
    
    print(f"   - Score volume: {dimension_metrics.get('volume_score', 0):.2f}/10")
    print(f"   - Variables: {dimension_metrics.get('variable_count', 0)}")
    print(f"   - Colonnes haute cardinalité: {dimension_metrics.get('high_cardinality_columns', 0)}")
    print(f"   - Tables: {dimension_metrics.get('table_count', 0)}")
    print(f"   - Score longitudinalité: {dimension_metrics.get('longitudinality_score', 0):.2f}")
    
    # 4. Visualisation des données
    print("\n4. Visualisation des données...")
    
    visualizer = MT5DVisualizer(style='seaborn')
    
    # Radar chart des dimensions
    fig1 = visualizer.plot_dimension_radar({
        'volume': {'score': dimension_metrics.get('volume_score', 0)},
        'many_variables': {'score': dimension_metrics.get('variable_count', 0) / 10},
        'high_cardinality': {'score': dimension_metrics.get('high_cardinality_columns', 0) / 5},
        'many_tables': {'score': dimension_metrics.get('table_count', 0) / 2},
        'repeated_measurements': {'score': dimension_metrics.get('longitudinality_score', 0) * 10}
    }, title="Profil 5D du Dataset")
    
    fig1.savefig('advanced_dimension_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Radar chart sauvegardé: advanced_dimension_analysis.png")
    
    # Matrice de corrélations
    fig2, corr_matrix = plot_correlation_matrix(tables)
    fig2.savefig('advanced_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✓ Matrice de corrélations sauvegardée: advanced_correlation_matrix.png")
    
    # 5. Pipeline avec configuration personnalisée
    print("\n5. Exécution du pipeline avec configuration personnalisée...")
    
    # Créer une configuration personnalisée
    custom_config = {
        'pipeline': {
            'name': 'advanced_medical_analysis',
            'output_dir': './advanced_results'
        },
        'profiling': {
            'detailed_analysis': True
        },
        'hypergraph': {
            'relation_types': ['explicit', 'temporal', 'semantic'],
            'construction': {
                'temporal_window_hours': 48,
                'use_hierarchical_clustering': True
            }
        },
        'embeddings': {
            'pente': {
                'output_dim': 512,
                'use_attention': True,
                'num_attention_heads': 16
            }
        },
        'models': {
            'rht': {
                'hidden_dim': 1024,
                'num_layers': 4,
                'dropout': 0.2
            }
        },
        'evaluation': {
            'metrics': {
                'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                'clustering': ['silhouette_score', 'adjusted_rand_score'],
                'temporal': ['mse_temporal', 'mae_temporal']
            }
        }
    }
    
    # Sauvegarder la configuration
    config_path = output_dir / 'custom_config.yaml'
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    # Exécuter le pipeline
    pipeline = MT5DPipeline(config_path=str(config_path))
    
    # Définir une tâche cible complexe
    target_task = {
        'name': 'patient_risk_prediction',
        'type': 'multi_task',
        'tasks': [
            {
                'name': 'readmission_prediction',
                'type': 'classification',
                'target': 'readmission_risk',
                'positive_class_weight': 2.0
            },
            {
                'name': 'length_of_stay_prediction',
                'type': 'regression',
                'target': 'length_of_stay_days'
            }
        ]
    }
    
    results = pipeline.run(
        tables=tables,
        relationships=relationships,
        target_task=target_task
    )
    
    # 6. Analyse des résultats
    print("\n6. Analyse approfondie des résultats...")
    
    # Calculer des métriques avancées
    metrics_calculator = MT5DMetrics()
    
    # Simuler des prédictions et vérités terrain pour l'exemple
    n_samples = 100
    y_pred_class = np.random.rand(n_samples, 2)
    y_true_class = np.random.randint(0, 2, n_samples)
    
    y_pred_reg = np.random.randn(n_samples)
    y_true_reg = np.random.randn(n_samples)
    
    clusters = np.random.randint(0, 5, n_samples)
    labels = np.random.randint(0, 5, n_samples)
    
    # Calculer toutes les métriques
    advanced_metrics = metrics_calculator.compute_all_metrics(
        predictions={
            'classification': y_pred_class,
            'regression': y_pred_reg,
            'clustering': clusters
        },
        ground_truth={
            'classification': y_true_class,
            'regression': y_true_reg,
            'clustering': labels
        },
        hypergraph=results.get('hypergraph')
    )
    
    # Créer un rapport détaillé
    report = metrics_calculator.create_summary_report()
    
    if not report.empty:
        print("\n   RAPPORT DÉTAILLÉ DES MÉTRIQUES:")
        print("   " + "-" * 60)
        
        for category in report['Category'].unique():
            cat_report = report[report['Category'] == category]
            print(f"\n   {category.upper()}:")
            for _, row in cat_report.iterrows():
                print(f"     {row['Metric']}: {row['Value']:.4f} - {row['Description']}")
    
    # 7. Visualisation des résultats
    print("\n7. Visualisation avancée des résultats...")
    
    # Espace d'embedding
    if 'embeddings' in results:
        embeddings = results['embeddings']
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().detach().numpy()
            
            # Réduire la dimension pour la visualisation
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings_np[:1000])  # Premier 1000
            
            fig3, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                alpha=0.6, s=20, c='blue')
            ax.set_title('Espace d\'Embedding (t-SNE)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.grid(True, alpha=0.3)
            
            fig3.savefig('advanced_embedding_space.png', dpi=300, bbox_inches='tight')
            print("   ✓ Espace d'embedding sauvegardé: advanced_embedding_space.png")
    
    # Historique des pertes
    if 'training_history' in results:
        history = results['training_history']
        
        fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot des pertes
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Évolution de la Perte')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot des métriques
        metrics_to_plot = ['accuracy', 'f1_score', 'roc_auc']
        for metric in metrics_to_plot:
            if metric in history:
                axes[1].plot(history[metric], label=metric, linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Évolution des Métriques')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig4.savefig('advanced_training_history.png', dpi=300, bbox_inches='tight')
        print("   ✓ Historique d'entraînement sauvegardé: advanced_training_history.png")
    
    # 8. Génération d'insights
    print("\n8. Génération d'insights avancés...")
    
    insights = generate_advanced_insights(results, tables, relationships)
    
    print("\n   INSIGHTS GÉNÉRÉS:")
    print("   " + "-" * 60)
    
    for i, insight in enumerate(insights[:10], 1):
        print(f"   {i}. {insight}")
    
    # 9. Sauvegarde complète
    print("\n9. Sauvegarde complète des résultats...")
    
    # Sauvegarder tous les résultats
    loader.save_results({
        'tables': tables,
        'relationships': relationships,
        'profiling_metrics': metrics,
        'dimension_metrics': dimension_metrics,
        'advanced_metrics': advanced_metrics,
        'results': results,
        'insights': insights,
        'correlation_matrix': corr_matrix
    }, './advanced_analysis_results')
    
    print("   ✓ Tous les résultats sauvegardés dans: advanced_analysis_results/")
    
    # 10. Création d'un rapport final
    print("\n10. Génération du rapport final...")
    
    generate_final_report(
        tables=tables,
        metrics=metrics,
        dimension_metrics=dimension_metrics,
        advanced_metrics=advanced_metrics,
        insights=insights,
        output_path='./advanced_final_report.md'
    )
    
    print("   ✓ Rapport final généré: advanced_final_report.md")
    
    print("\n" + "=" * 80)
    print("ANALYSE AVANCÉE TERMINÉE AVEC SUCCÈS!")
    print("=" * 80)

def generate_advanced_insights(results, tables, relationships):
    """Génère des insights avancés à partir des résultats"""
    
    insights = []
    
    # Insights basés sur le profilage
    total_rows = sum(len(df) for df in tables.values())
    total_columns = sum(len(df.columns) for df in tables.values())
    
    insights.append(
        f"Dataset analysé: {len(tables)} tables, {total_rows:,} lignes, "
        f"{total_columns} colonnes, {len(relationships)} relations"
    )
    
    # Insights basés sur les relations
    if relationships:
        relation_types = {}
        for rel in relationships:
            rel_type = rel[4] if len(rel) > 4 else 'related_to'
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        insights.append(
            f"Types de relations: {', '.join([f'{k}: {v}' for k, v in relation_types.items()])}"
        )
    
    # Insights basés sur les métriques
    if 'advanced_metrics' in results:
        metrics = results['advanced_metrics']
        
        if 'accuracy' in metrics and metrics['accuracy'] > 0.8:
            insights.append(
                f"Modèle performant: accuracy de {metrics['accuracy']:.3f} "
                f"(précision: {metrics.get('precision', 0):.3f}, "
                f"rappel: {metrics.get('recall', 0):.3f})"
            )
        
        if 'silhouette_score' in metrics:
            score = metrics['silhouette_score']
            if score > 0.5:
                insights.append(
                    f"Clusters bien définis: score de silhouette de {score:.3f} "
                    f"({metrics.get('n_clusters', 0)} clusters détectés)"
                )
    
    # Insights temporels
    if 'visits' in tables and 'visit_date' in tables['visits'].columns:
        visits_df = tables['visits']
        visits_df['visit_date'] = pd.to_datetime(visits_df['visit_date'])
        
        # Nombre de visites par mois
        visits_by_month = visits_df.set_index('visit_date').resample('M').size()
        if len(visits_by_month) > 1:
            growth_rate = (visits_by_month.iloc[-1] / visits_by_month.iloc[0] - 1) * 100
            insights.append(
                f"Tendance temporelle: {growth_rate:+.1f}% de croissance "
                f"sur {len(visits_by_month)} mois"
            )
    
    # Insights médicaux spécifiques
    if 'patients' in tables and 'diagnoses' in tables:
        patients_df = tables['patients']
        diagnoses_df = tables['diagnoses']
        
        # Âge moyen des patients
        avg_age = patients_df['age'].mean()
        insights.append(f"Âge moyen des patients: {avg_age:.1f} ans")
        
        # Diagnostiques les plus fréquents
        if 'code' in diagnoses_df.columns:
            top_diagnoses = diagnoses_df['code'].value_counts().head(3)
            insights.append(
                f"Diagnostiques les plus fréquents: {', '.join(top_diagnoses.index.tolist())}"
            )
    
    # Insights sur la qualité des données
    missing_percentages = []
    for table_name, df in tables.items():
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        missing_percentages.append((table_name, missing_pct))
    
    high_missing = [(name, pct) for name, pct in missing_percentages if pct > 5]
    if high_missing:
        insights.append(
            f"Tables avec >5% valeurs manquantes: "
            f"{', '.join([f'{name} ({pct:.1f}%)' for name, pct in high_missing])}"
        )
    
    return insights

def generate_final_report(tables, metrics, dimension_metrics, 
                         advanced_metrics, insights, output_path):
    """Génère un rapport Markdown final"""
    
    report = f"""# Rapport d'Analyse MT5D - Analyse Avancée

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Résumé Exécutif

Cette analyse utilise le framework MT5D pour analyser un dataset multitable complexe.

### Points Clés:
- Dataset: {len(tables)} tables, {sum(len(df) for df in tables.values()):,} lignes
- Métriques dimensionnelles calculées
- Modèles avancés appliqués
- Insights business générés

## 2. Profilage des Données

### 2.1 Métriques Dimensionnelles

| Dimension | Score | Détails |
|-----------|-------|---------|
"""

    # Ajouter les métriques dimensionnelles
    dim_mapping = {
        'volume_score': ('Volume', '/10'),
        'variable_count': ('Variables', ' colonnes'),
        'high_cardinality_columns': ('Haute Cardinalité', ' colonnes'),
        'table_count': ('Tables', ' tables'),
        'longitudinality_score': ('Mesures Répétées', '/1')
    }
    
    for metric_key, (dim_name, unit) in dim_mapping.items():
        if metric_key in dimension_metrics:
            value = dimension_metrics[metric_key]
            report += f"| {dim_name} | {value:.2f}{unit} | \n"
    
    report += """

### 2.2 Statistiques par Table

| Table | Lignes | Colonnes | Valeurs Manquantes |
|-------|--------|----------|-------------------|
"""
    
    # Ajouter les stats par table
    for table_name, df in tables.items():
        n_rows = len(df)
        n_cols = len(df.columns)
        missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100
        
        report += f"| {table_name} | {n_rows:,} | {n_cols} | {missing_pct:.1f}% |\n"
    
    report += """
## 3. Résultats d'Analyse

### 3.1 Métriques Avancées
"""
    
    # Grouper les métriques par catégorie
    metric_categories = {
        'Classification': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
        'Régression': ['mse', 'rmse', 'mae', 'r2_score'],
        'Clustering': ['silhouette_score', 'adjusted_rand_score', 'n_clusters'],
        'Relationnel': ['num_nodes', 'num_edges', 'edge_density']
    }
    
    for category, metric_list in metric_categories.items():
        report += f"\n#### {category}\n\n"
        
        for metric in metric_list:
            if metric in advanced_metrics:
                value = advanced_metrics[metric]
                if isinstance(value, float):
                    report += f"- **{metric}**: {value:.4f}\n"
                else:
                    report += f"- **{metric}**: {value}\n"
    
    report += """
## 4. Insights Business

### 4.1 Insights Générés
"""
    
    for i, insight in enumerate(insights, 1):
        report += f"{i}. {insight}\n"
    
    report += """
## 5. Recommandations

### 5.1 Pour l'Amélioration des Données

1. **Nettoyage des données**: Adresser les valeurs manquantes identifiées
2. **Standardisation**: Harmoniser les formats de dates et codes
3. **Documentation**: Documenter les relations entre tables

### 5.2 Pour l'Analyse Future

1. **Intégration de nouvelles sources**: Ajouter des données externes
2. **Analyse temporelle approfondie**: Étudier les tendances saisonnières
3. **Modèles prédictifs**: Développer des modèles pour la prévention

## 6. Annexes

### 6.1 Visualisations Générées

Les visualisations suivantes ont été générées:
1. `advanced_dimension_analysis.png` - Radar chart des dimensions
2. `advanced_correlation_matrix.png` - Matrice de corrélations
3. `advanced_embedding_space.png` - Espace d'embedding
4. `advanced_training_history.png` - Historique d'entraînement

### 6.2 Fichiers de Résultats

Tous les résultats détaillés sont disponibles dans `advanced_analysis_results/`

---

*Rapport généré automatiquement par le framework MT5D*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    advanced_analysis_example()
