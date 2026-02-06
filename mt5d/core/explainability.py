"""
Explainability Module
Manuscrit Section 9.2: "Interpretability: Preservation of relational structure"
"""
import torch
import dgl
import pandas as pd
from typing import List, Tuple, Dict

class RelationalExplainer:
    """
    Explique les prédictions du modèle RHT en analysant les poids d'attention
    """
    def __init__(self, model: torch.nn.Module, hypergraph: dgl.DGLGraph):
        self.model = model
        self.graph = hypergraph

    def explain_node(self, node_idx: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Identifie les nœuds et relations les plus influents pour un nœud donné
        """
        self.model.eval()
        
        # Extraction des poids d'attention du modèle (si stockés dans le graphe lors du forward)
        # On suppose que le modèle a sauvegardé les 'attention_weights' dans edata
        if 'a' not in self.graph.edata:
            return {"error": "Attention weights not found. Run forward pass first."}
            
        # Récupérer les arêtes entrantes vers le nœud cible
        in_edges = self.graph.in_edges(node_idx, form='eid')
        
        if len(in_edges) == 0:
            return {"info": "Isolated node"}

        # Récupérer les poids d'attention associés
        attn_weights = self.graph.edata['a'][in_edges].squeeze()
        
        # Trouver les top-k contributeurs
        if attn_weights.ndim == 0: # Cas d'une seule arête
             top_indices = [0]
        else:
            k = min(top_k, len(attn_weights))
            top_values, top_indices = torch.topk(attn_weights, k)
            top_indices = top_indices.tolist()

        # Mapper les indices d'arêtes vers les nœuds sources (tables d'origine)
        src_nodes, _, _ = self.graph.find_edges(in_edges[top_indices])
        
        explanations = []
        for src_idx, weight in zip(src_nodes, top_values if attn_weights.ndim > 0 else [attn_weights]):
            src_idx = int(src_idx)
            # Récupérer métadonnées si disponibles
            # node_info = self.graph.ndata['info'][src_idx] 
            explanations.append({
                "source_node_idx": src_idx,
                "importance_score": float(weight),
                "relationship_type": "implicit/explicit" # À récupérer du graphe
            })
            
        return {
            "target_node": node_idx,
            "top_contributors": explanations
        }

    def generate_explanation_report(self, node_indices: List[int]) -> pd.DataFrame:
        """Génère un rapport d'explicabilité pour un ensemble de nœuds"""
        rows = []
        for idx in node_indices:
            expl = self.explain_node(idx, top_k=1)
            if "top_contributors" in expl:
                top = expl["top_contributors"][0]
                rows.append({
                    "Node ID": idx,
                    "Top Contributor ID": top["source_node_idx"],
                    "Importance": top["importance_score"]
                })
        return pd.DataFrame(rows)
