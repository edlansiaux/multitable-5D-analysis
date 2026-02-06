import torch
import dgl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class RelationalHypergraphBuilder:
    """
    Implémente l'Algorithme 1 : Adaptive Hypergraph Construction.
    Transforme les tables relationnelles en un graphe bipartite (Entités <-> Hyperarêtes).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.node_dim = self.config.get('node_dim', 128)

    def build_from_tables(self, 
                         tables: Dict[str, pd.DataFrame],
                         relationships: List[Tuple[str, str, str, str]],
                         temporal_columns: Dict[str, str] = None) -> dgl.DGLGraph:
        """
        Construit l'hypergraphe.
        relationships format: [(table_src, col_src, table_tgt, col_tgt, type), ...]
        """
        print(f"Construction de l'hypergraphe à partir de {len(tables)} tables...")
        
        # 1. Création des nœuds d'Entité
        # Mapping: (table_name, row_idx) -> global_id
        entity_mapping = {} 
        entity_features = []
        global_id_counter = 0
        
        # Stocker les métadonnées pour reconstruire les tables plus tard si besoin
        self.node_info = []

        for table_name, df in tables.items():
            # Conversion basique des features numériques
            numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
            feats = numeric_df.values
            
            for local_idx in range(len(df)):
                entity_mapping[(table_name, local_idx)] = global_id_counter
                self.node_info.append({'table': table_name, 'local_idx': local_idx})
                
                # Padding/Truncate features
                f_vec = np.zeros(self.node_dim)
                curr_f = feats[local_idx]
                dim = min(len(curr_f), self.node_dim)
                f_vec[:dim] = curr_f[:dim]
                entity_features.append(f_vec)
                
                global_id_counter += 1
                
        num_entities = global_id_counter
        print(f"  - {num_entities} entités créées.")

        # 2. Création des Hyperarêtes (Nœuds de type 'hyperedge' dans le graphe bipartite)
        # Pour simplifier, chaque relation FK crée une hyperarête binaire, 
        # mais la structure permet des n-aires.
        
        src_nodes = [] # Entités
        dst_nodes = [] # Hyperarêtes
        he_features = [] # Features des hyperarêtes
        
        he_id_counter = 0
        
        # Traitement des FKs (Relations Explicites)
        for rel in relationships:
            t_src, c_src, t_tgt, c_tgt, rel_type = rel
            
            if t_src not in tables or t_tgt not in tables:
                continue
                
            # Jointure pour trouver les connexions
            df_src = tables[t_src].reset_index()
            df_tgt = tables[t_tgt].reset_index()
            
            # On suppose que c_src et c_tgt contiennent les clés
            merged = pd.merge(df_src, df_tgt, left_on=c_src, right_on=c_tgt, suffixes=('_s', '_t'))
            
            # Création des arêtes du graphe bipartite
            for _, row in merged.iterrows():
                idx_src = row['index_s'] # index original pandas (si reset_index préserve l'ordre 0..N)
                idx_tgt = row['index_t']
                
                # IDs globaux des entités
                u = entity_mapping.get((t_src, idx_src))
                v = entity_mapping.get((t_tgt, idx_tgt))
                
                if u is not None and v is not None:
                    # Créer un nœud Hyperarête H_i
                    he_id = he_id_counter
                    
                    # Connecter u -> H_i et v -> H_i
                    # Dans DGL bipartite: (u, v) où u type A, v type B
                    src_nodes.extend([u, v])
                    dst_nodes.extend([he_id, he_id])
                    
                    # Feature de l'hyperarête (ex: one-hot encoding du type de relation)
                    he_vec = np.zeros(64) # Dim arbitraire
                    # he_vec[type_id] = 1 ...
                    he_features.append(he_vec)
                    
                    he_id_counter += 1

        print(f"  - {he_id_counter} hyperarêtes relationnelles détectées.")

        # 3. Construction DGL (Hétérogène Bipartite)
        # Graph data: { ('entity', 'in', 'hyperedge'): (u, v), ('hyperedge', 'con', 'entity'): (v, u) }
        # Pour le RHT, on veut souvent faire passer les messages Entité -> Hyperarête -> Entité
        
        data_dict = {
            ('entity', 'part_of', 'hyperedge'): (torch.tensor(src_nodes).long(), torch.tensor(dst_nodes).long()),
            ('hyperedge', 'contains', 'entity'): (torch.tensor(dst_nodes).long(), torch.tensor(src_nodes).long())
        }
        
        g = dgl.heterograph(data_dict)
        
        # Attacher les features
        g.nodes['entity'].data['feat'] = torch.FloatTensor(np.array(entity_features))
        if he_features:
            g.nodes['hyperedge'].data['feat'] = torch.FloatTensor(np.array(he_features))
        else:
            g.nodes['hyperedge'].data['feat'] = torch.zeros((g.num_nodes('hyperedge'), 64))
            
        return g
