"""
Step 7: Operationalization and Monitoring
Manuscrit Section 5.8
"""
import numpy as np
import dgl
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict, Any

class RelationalDriftMonitor:
    """
    Détecte les changements dans la structure relationnelle et les données
    """
    def __init__(self, reference_graph: dgl.DGLGraph, config: Dict[str, Any]):
        self.ref_graph = reference_graph
        self.config = config
        self.ref_stats = self._compute_graph_stats(reference_graph)
        
    def check_drift(self, new_graph: dgl.DGLGraph) -> Dict[str, Any]:
        """
        Vérifie si le nouveau graphe dévie significativement de la référence
        """
        print("Step 7: Monitoring - Checking for Relational Drift...")
        current_stats = self._compute_graph_stats(new_graph)
        drift_report = {}
        alert = False

        # 1. Dérive Structurelle (Topologie)
        density_change = abs(current_stats['density'] - self.ref_stats['density'])
        if density_change > self.config.get('density_threshold', 0.05):
            drift_report['structure'] = f"Significant density change: {density_change:.4f}"
            alert = True

        # 2. Dérive Relationnelle (Distribution des arêtes)
        # Comparaison des distributions de degrés via Wasserstein
        deg_dist = wasserstein_distance(self.ref_stats['degrees'], current_stats['degrees'])
        if deg_dist > self.config.get('degree_threshold', 1.0):
            drift_report['topology'] = f"Degree distribution shift: {deg_dist:.4f}"
            alert = True

        # 3. Dérive Sémantique (Embeddings/Attributs)
        # Test Kolmogorov-Smirnov sur les attributs principaux (si disponibles)
        # Ici simplifié
        pass

        drift_report['alert_triggered'] = alert
        return drift_report

    def _compute_graph_stats(self, g: dgl.DGLGraph) -> Dict[str, Any]:
        """Calcule les statistiques clés du graphe pour le monitoring"""
        num_nodes = g.num_nodes()
        num_edges = g.num_edges()
        
        in_degrees = g.in_degrees().float().cpu().numpy()
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            'degrees': in_degrees,
            'avg_degree': np.mean(in_degrees)
        }
