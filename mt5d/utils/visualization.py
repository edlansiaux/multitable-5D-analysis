"""
Module de visualisation pour l'analyse multitable 5D
"""
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

class MT5DVisualizer:
    """Visualiseur pour les résultats MT5D"""
    
    def __init__(self, style: str = "seaborn"):
        self.style = style
        plt.style.use(style)
        
    def plot_dimension_radar(self, metrics: Dict[str, Dict[str, float]], 
                            title: str = "Analyse 5D des Données"):
        """Crée un radar chart pour les métriques 5D"""
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Catégories
        categories = ['Volume', 'Many Variables', 'High Cardinality', 
                     'Many Tables', 'Repeated Measurements']
        
        # Valeurs normalisées (0-10)
        values = [
            self._normalize_metric(metrics.get('volume', {}).get('score', 0)),
            self._normalize_metric(metrics.get('many_variables', {}).get('score', 0)),
            self._normalize_metric(metrics.get('high_cardinality', {}).get('score', 0)),
            self._normalize_metric(metrics.get('many_tables', {}).get('score', 0)),
            self._normalize_metric(metrics.get('repeated_measurements', {}).get('score', 0))
        ]
        
        # Compléter le cercle
        values += values[:1]
        
        # Angles
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, linewidth=2, linestyle='solid', label='Scores')
        ax.fill(angles, values, alpha=0.25)
        
        # Axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 10)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Title
        plt.title(title, size=20, y=1.1)
        
        return fig
    
    def plot_hypergraph(self, hypergraph, layout: str = 'spring', 
                       node_size: int = 500, figsize: tuple = (12, 10)):
        """Visualise un hypergraphe DGL"""
        
        # Convertir DGL en NetworkX pour visualisation
        g_nx = self._dgl_to_networkx(hypergraph)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(g_nx)
        elif layout == 'circular':
            pos = nx.circular_layout(g_nx)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(g_nx)
        else:
            pos = nx.random_layout(g_nx)
        
        # Couleurs par type de nœud
        node_colors = []
        for node in g_nx.nodes():
            if 'patient' in str(node).lower():
                node_colors.append('#FF6B6B')  # Rouge pour patients
            elif 'visit' in str(node).lower():
                node_colors.append('#4ECDC4')  # Turquoise pour visites
            elif 'diagnosis' in str(node).lower():
                node_colors.append('#45B7D1')  # Bleu pour diagnostics
            else:
                node_colors.append('#96CEB4')  # Vert pour autres
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(g_nx, pos, node_size=node_size, 
                              node_color=node_colors, alpha=0.8, ax=ax)
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(g_nx, pos, alpha=0.5, width=1, ax=ax)
        
        # Labels
        nx.draw_networkx_labels(g_nx, pos, font_size=8, ax=ax)
        
        # Title
        ax.set_title("Hypergraphe Relationnel", fontsize=16)
        ax.axis('off')
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Patients', alpha=0.8),
            Patch(facecolor='#4ECDC4', label='Visites', alpha=0.8),
            Patch(facecolor='#45B7D1', label='Diagnostics', alpha=0.8),
            Patch(facecolor='#96CEB4', label='Autres', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
    
    def plot_temporal_patterns(self, temporal_data: pd.DataFrame, 
                              entity_id: str = None, 
                              figsize: tuple = (14, 8)):
        """Visualise les patterns temporels"""
        
        if entity_id:
            data = temporal_data[temporal_data['entity_id'] == entity_id]
            title = f"Patterns Temporels - {entity_id}"
        else:
            data = temporal_data
            title = "Patterns Temporels - Toutes les entités"
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Distribution des fréquences
        if 'frequency' in data.columns:
            axes[0].hist(data['frequency'].dropna(), bins=30, alpha=0.7, color='#3498DB')
            axes[0].set_title('Distribution des Fréquences')
            axes[0].set_xlabel('Fréquence')
            axes[0].set_ylabel('Count')
        
        # 2. Série temporelle
        if 'timestamp' in data.columns and 'value' in data.columns:
            sample_data = data.sample(min(100, len(data)))
            axes[1].scatter(sample_data['timestamp'], sample_data['value'], 
                          alpha=0.6, color='#E74C3C')
            axes[1].set_title('Série Temporelle (échantillon)')
            axes[1].set_xlabel('Timestamp')
            axes[1].set_ylabel('Valeur')
            axes[1].tick_params(axis='x', rotation=45)
        
        # 3. Box plot par catégorie
        if 'category' in data.columns and 'value' in data.columns:
            categories = data['category'].value_counts().index[:5]
            filtered_data = data[data['category'].isin(categories)]
            
            box_data = []
            for cat in categories:
                box_data.append(filtered_data[filtered_data['category'] == cat]['value'].values)
            
            axes[2].boxplot(box_data, labels=categories)
            axes[2].set_title('Distribution par Catégorie')
            axes[2].set_xlabel('Catégorie')
            axes[2].set_ylabel('Valeur')
            axes[2].tick_params(axis='x', rotation=45)
        
        # 4. Heatmap de corrélations temporelles
        if len(data) > 10 and 'value' in data.columns:
            # Créer une matrice de corrélations (exemple simplifié)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                im = axes[3].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                axes[3].set_title('Corrélations Temporelles')
                axes[3].set_xticks(range(len(numeric_cols)))
                axes[3].set_xticklabels(numeric_cols, rotation=45)
                axes[3].set_yticks(range(len(numeric_cols)))
                axes[3].set_yticklabels(numeric_cols)
                plt.colorbar(im, ax=axes[3])
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        return fig
    
    def plot_embedding_space(self, embeddings: np.ndarray, 
                            labels: np.ndarray = None,
                            method: str = 'tsne',
                            figsize: tuple = (10, 8)):
        """Visualise l'espace d'embedding avec réduction de dimension"""
        
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Réduction de dimension
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Méthode {method} non supportée")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            # Scatter plot coloré par labels
            unique_labels = np.unique(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                          c=[color], label=str(label), alpha=0.6, s=50)
            
            ax.legend(title='Labels')
        else:
            # Scatter plot simple
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50, 
                      c='#2E86AB')
        
        ax.set_title(f'Espace d\'Embedding ({method.upper()})', fontsize=16)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_interactive_dashboard(self, results: Dict[str, Any]):
        """Crée un dashboard interactif Plotly"""
        
        fig = go.Figure()
        
        # Ajouter plusieurs visualisations
        # 1. Radar chart interactif
        if 'dimension_metrics' in results:
            self._add_interactive_radar(fig, results['dimension_metrics'])
        
        # 2. Graphique de relations
        if 'relationship_graph' in results:
            self._add_interactive_graph(fig, results['relationship_graph'])
        
        # 3. Timeline des insights
        if 'insights' in results:
            self._add_insights_timeline(fig, results['insights'])
        
        fig.update_layout(
            title="Dashboard MT5D - Analyse Multitable",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def _normalize_metric(self, value: float, max_val: float = 100) -> float:
        """Normalise une métrique entre 0 et 10"""
        return min(10, (value / max_val) * 10) if max_val > 0 else 0
    
    def _dgl_to_networkx(self, dgl_graph):
        """Convertit un graphe DGL en NetworkX (simplifié)"""
        import networkx as nx
        
        g_nx = nx.Graph()
        
        # Ajouter les nœuds
        num_nodes = dgl_graph.num_nodes()
        for i in range(num_nodes):
            g_nx.add_node(f"node_{i}")
        
        # Ajouter les arêtes (simplifié)
        # Note: Cette conversion est basique et peut être améliorée
        src, dst = dgl_graph.edges()
        for s, d in zip(src.numpy(), dst.numpy()):
            g_nx.add_edge(f"node_{s}", f"node_{d}")
        
        return g_nx
    
    def _add_interactive_radar(self, fig, metrics):
        """Ajoute un radar chart interactif"""
        # Implémentation simplifiée
        pass
    
    def _add_interactive_graph(self, fig, graph_data):
        """Ajoute un graphe interactif"""
        # Implémentation simplifiée
        pass
    
    def _add_insights_timeline(self, fig, insights):
        """Ajoute une timeline des insights"""
        # Implémentation simplifiée
        pass

def plot_correlation_matrix(tables: Dict[str, pd.DataFrame], 
                          method: str = 'pearson',
                          figsize: tuple = (14, 12)):
    """Crée une matrice de corrélations entre tables"""
    
    # Extraire toutes les colonnes numériques
    all_numeric = []
    column_mapping = {}
    
    for table_name, df in tables.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_name = f"{table_name}.{col}"
            all_numeric.append(df[col].values)
            column_mapping[col_name] = (table_name, col)
    
    # Créer la matrice de données
    data_matrix = np.column_stack(all_numeric)
    column_names = list(column_mapping.keys())
    
    # Calculer la corrélation
    corr_matrix = pd.DataFrame(data_matrix, columns=column_names).corr(method=method)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Créer une colormap personnalisée
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', ['#E74C3C', '#FFFFFF', '#3498DB']
    )
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(column_names)))
    ax.set_yticks(np.arange(len(column_names)))
    ax.set_xticklabels(column_names, rotation=90, ha='right')
    ax.set_yticklabels(column_names)
    
    # Grid
    ax.set_xticks(np.arange(len(column_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(column_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Corrélation', rotation=-90, va='bottom')
    
    # Title
    ax.set_title('Matrice de Corrélations Inter-Tables', fontsize=16, pad=20)
    
    # Annoter les valeurs
    threshold = 0.7
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            value = corr_matrix.iloc[i, j]
            if abs(value) > threshold and i != j:
                text_color = 'white' if abs(value) > 0.8 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    plt.tight_layout()
    return fig, corr_matrix
