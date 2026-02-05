import torch
import dgl
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HyperEdge:
    """Représentation d'une hyperarête"""
    id: str
    nodes: List[str]  # IDs des nœuds
    type: str  # Type de relation
    weight: float = 1.0
    attributes: Dict = None
    temporal_info: Optional[Dict] = None

class RelationalHypergraphBuilder:
    """Constructeur d'hypergraphes relationnels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hypergraph = None
        self.node_features = {}
        self.edge_features = {}
        
    def build_from_tables(self, 
                         tables: Dict[str, pd.DataFrame],
                         relationships: List[Tuple],
                         temporal_columns: Dict[str, List[str]] = None):
        """Construit l'hypergraphe à partir de tables"""
        
        # Étape 1: Création des nœuds
        nodes = self._create_nodes(tables)
        
        # Étape 2: Création des hyperarêtes basées sur les relations
        hyperedges = self._create_hyperedges_from_relationships(
            tables, nodes, relationships
        )
        
        # Étape 3: Découverte de relations implicites
        implicit_hyperedges = self._discover_implicit_relations(
            tables, nodes
        )
        
        # Étape 4: Intégration temporelle
        if temporal_columns:
            self._integrate_temporal_information(
                tables, nodes, hyperedges, temporal_columns
            )
        
        # Étape 5: Construction du graphe DGL
        self.hypergraph = self._build_dgl_hypergraph(nodes, hyperedges)
        
        return self.hypergraph
    
    def _create_nodes(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Crée les nœuds à partir des tables"""
        nodes = {}
        
        for table_name, df in tables.items():
            for idx, row in df.iterrows():
                node_id = f"{table_name}_{idx}"
                
                # Features du nœud
                features = {}
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        features[f"{col}"] = float(row[col])
                    elif pd.api.types.is_categorical_dtype(df[col]) or \
                         pd.api.types.is_object_dtype(df[col]):
                        # Pour high-cardinality, on utilise un embedding initial
                        features[f"{col}_cat"] = hash(str(row[col])) % 1000
                
                nodes[node_id] = {
                    "features": features,
                    "type": table_name,
                    "original_index": idx,
                    "table": table_name,
                }
        
        return nodes
    
    def _create_hyperedges_from_relationships(self, 
                                             tables: Dict[str, pd.DataFrame],
                                             nodes: Dict[str, Dict],
                                             relationships: List[Tuple]):
        """Crée des hyperarêtes à partir des relations connues"""
        hyperedges = []
        
        for rel in relationships:
            src_table, src_col, tgt_table, tgt_col, rel_type = rel
            
            # Mapping valeur -> nœuds pour chaque table
            src_mapping = defaultdict(list)
            for node_id, node_info in nodes.items():
                if node_info["table"] == src_table:
                    row_idx = node_info["original_index"]
                    value = tables[src_table].loc[row_idx, src_col]
                    src_mapping[value].append(node_id)
            
            tgt_mapping = defaultdict(list)
            for node_id, node_info in nodes.items():
                if node_info["table"] == tgt_table:
                    row_idx = node_info["original_index"]
                    value = tables[tgt_table].loc[row_idx, tgt_col]
                    tgt_mapping[value].append(node_id)
            
            # Création d'hyperarêtes pour les valeurs communes
            common_values = set(src_mapping.keys()) & set(tgt_mapping.keys())
            
            for value in common_values:
                connected_nodes = src_mapping[value] + tgt_mapping[value]
                
                hyperedge = HyperEdge(
                    id=f"{rel_type}_{value}",
                    nodes=connected_nodes,
                    type=rel_type,
                    weight=1.0,
                    attributes={"value": value}
                )
                hyperedges.append(hyperedge)
        
        return hyperedges
    
    def _discover_implicit_relations(self, 
                                    tables: Dict[str, pd.DataFrame],
                                    nodes: Dict[str, Dict]):
        """Découvre des relations implicites via similarité"""
        hyperedges = []
        
        # Pour chaque table, chercher des similarités entre colonnes
        for table_name, df in tables.items():
            # Utiliser des embeddings pour les colonnes catégorielles
            categorical_cols = df.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            
            for col in categorical_cols:
                # Group by value et créer des hyperarêtes pour les groupes
                value_to_nodes = defaultdict(list)
                
                for node_id, node_info in nodes.items():
                    if node_info["table"] == table_name:
                        row_idx = node_info["original_index"]
                        value = df.loc[row_idx, col]
                        value_to_nodes[value].append(node_id)
                
                # Créer une hyperarête pour chaque groupe de valeur
                for value, node_list in value_to_nodes.items():
                    if len(node_list) > 1:  # Au moins 2 nœuds
                        hyperedge = HyperEdge(
                            id=f"implicit_{table_name}_{col}_{value}",
                            nodes=node_list,
                            type=f"implicit_{col}",
                            weight=0.5,  # Poids plus faible pour relations implicites
                            attributes={"column": col, "value": value}
                        )
                        hyperedges.append(hyperedge)
        
        return hyperedges
    
    def _build_dgl_hypergraph(self, nodes: Dict[str, Dict], 
                             hyperedges: List[HyperEdge]):
        """Construit un hypergraphe DGL"""
        
        # Créer un mapping nœud ID -> index
        node_ids = list(nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Extraire les features des nœuds
        node_features_list = []
        for node_id in node_ids:
            features = nodes[node_id]["features"]
            # Convertir en tensor
            feat_tensor = torch.tensor(list(features.values()), dtype=torch.float)
            node_features_list.append(feat_tensor)
        
        node_features = torch.stack(node_features_list)
        
        # Construire l'hypergraphe
        hyperedge_nodes = []
        hyperedge_types = []
        hyperedge_weights = []
        
        for hyperedge in hyperedges:
            node_indices = [node_to_idx[node_id] for node_id in hyperedge.nodes]
            hyperedge_nodes.append(node_indices)
            hyperedge_types.append(hyperedge.type)
            hyperedge_weights.append(hyperedge.weight)
        
        # Créer le graphe DGL
        g = dgl.heterograph({
            ('node', 'in', 'hyperedge'): (
                torch.cat([torch.tensor(nodes) for nodes in hyperedge_nodes]),
                torch.arange(len(hyperedges)).repeat_interleave(
                    torch.tensor([len(nodes) for nodes in hyperedge_nodes])
                )
            )
        })
        
        # Ajouter les features
        g.nodes['node'].data['feat'] = node_features
        g.edges['in'].data['weight'] = torch.tensor(hyperedge_weights)
        g.edges['in'].data['type'] = torch.tensor(
            [self._type_to_idx[t] for t in hyperedge_types]
        )
        
        return g
