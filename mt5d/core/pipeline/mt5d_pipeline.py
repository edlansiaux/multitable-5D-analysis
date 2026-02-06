import torch
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from ..profiling.dimensional_profiler import DimensionalProfiler
from ..hypergraph.builder import RelationalHypergraphBuilder
from ...models.architectures.rht import RelationalHypergraphTransformer

class MT5DPipeline:
    """
    Pipeline complet d'analyse multitable 5D (Sections 5.1 à 5.8)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.profiler = DimensionalProfiler(self.config.get('profiling', {}))
        self.builder = RelationalHypergraphBuilder(self.config.get('hypergraph', {}))
        self.model = None
        
    def _load_config(self, path):
        if path:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}
        
    def run(self, 
            tables: Dict[str, pd.DataFrame],
            relationships: List[tuple],
            target_task: str = "classification") -> Dict[str, Any]:
        
        print("=== MT5D Pipeline ===")
        
        # Step 0: Profilage
        metrics = self.profiler.profile(tables, relationships)
        
        # Step 1: Construction Hypergraphe
        # Utilisation de la colonne temporelle détectée pour guider la construction si besoin
        hypergraph = self.builder.build_from_tables(tables, relationships)
        
        # Step 4: Init Modèle (RHT)
        # On détermine les dims d'input dynamiquement basées sur le profilage
        input_dim = 128 # Défini dans le builder par défaut
        self.model = RelationalHypergraphTransformer(
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=10 if target_task == 'classification' else 1,
            use_pente=True
        )
        
        print(f"Modèle RHT initialisé avec {hypergraph.num_nodes()} nœuds.")
        
        # Simulation d'exécution (Forward pass simple pour vérifier que ça tourne)
        # Dans un vrai cas, ici il y aurait la boucle d'entraînement (Step 3 & 4 du papier)
        with torch.no_grad():
            # Extraction features factices pour la démo
            node_feats = hypergraph.nodes['entity'].data['feat']
            
            # Note: Le RHT défini dans rht.py attend un graphe homogène ou doit être adapté pour hétérogène.
            # Pour ce script de démo, on suppose une conversion interne ou une simplification dans RHT.
            # Ici, pour éviter l'erreur, on convertit le graphe bipartite en homogène pour le RHT actuel
            g_homo = dgl.to_homogeneous(hypergraph, ndata=['feat'])
            feats = g_homo.ndata['feat']
            
            output = self.model(g_homo, feats)
            print("Inférence test réussie. Shape sortie:", output.shape)
            
        return {"status": "success", "model": self.model, "graph": hypergraph}

    def save_pipeline(self, output_dir: str):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.model:
            torch.save(self.model.state_dict(), path / "rht_model.pt")
        print(f"Pipeline sauvegardé dans {output_dir}")
