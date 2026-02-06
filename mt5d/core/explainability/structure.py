import torch
import dgl
import networkx as nx
from typing import Dict, List, Any

class RelationalExplainer:
    """
    Module d'explicabilité (Section 5.8).
    Extrait les sous-graphes pertinents et les relations fortes basées sur l'attention.
    """
    
    def __init__(self, model):
        self.model = model
        
    def extract_important_subgraph(self, 
                                 graph: dgl.DGLGraph, 
                                 edge_weights: torch.Tensor, 
                                 threshold: float = 0.5) -> List[Dict]:
        """
        Identifie les relations critiques pour la prédiction (ex: Diagnostique X -> Labo Y).
        """
        print("Extraction des motifs relationnels explicatifs...")
        
        # Récupération des indices des arêtes importantes
        important_edges_idx = (edge_weights > threshold).nonzero(as_tuple=True)[0]
        
        explanations = []
        
        # Pour récupérer les IDs des nœuds, on a besoin du graphe original ou des métadonnées
        # Ici on suppose que le graphe a des attributs 'global_id' ou similaire
        src_ids, dst_ids = graph.find_edges(important_edges_idx)
        
        src_ids = src_ids.cpu().numpy()
        dst_ids = dst_ids.cpu().numpy()
        weights = edge_weights[important_edges_idx].detach().cpu().numpy()
        
        for s, d, w in zip(src_ids, dst_ids, weights):
            explanations.append({
                "source_node_idx": int(s),
                "target_node_idx": int(d),
                "importance_score": float(w),
                "relation_type": "inferred_strong_connection"
            })
            
        return explanations

    def visualize_attention(self, graph, edge_weights, output_path="attention_graph.html"):
        """
        Génère une visualisation interactive du graphe pondéré par l'attention.
        """
        # Conversion vers NetworkX pour visualisation simple
        # Dans un vrai dashboard, on exporterait vers JSON pour D3.js ou Cytoscape
        pass
