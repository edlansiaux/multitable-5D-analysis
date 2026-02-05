from typing import Dict, Any, Optional
import yaml
from pathlib import Path

class MT5DPipeline:
    """
    Pipeline complet d'analyse multitable 5D
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.profiler = DimensionalProfiler(self.config.get('profiling', {}))
        self.builder = RelationalHypergraphBuilder(self.config.get('hypergraph', {}))
        self.model = None
        self.results = {}
        
    def run(self, 
            tables: Dict[str, pd.DataFrame],
            relationships: Optional[List] = None,
            target_task: Optional[str] = None) -> Dict[str, Any]:
        """
        Exécute le pipeline complet
        """
        
        print("=== MT5D Pipeline - Démarrage ===")
        
        # Étape 1: Profilage
        print("Étape 1: Profilage 5D...")
        metrics = self.profiler.profile(tables, relationships)
        recommendations = self.profiler.recommend_pipeline()
        
        # Étape 2: Construction d'hypergraphe
        print("Étape 2: Construction d'hypergraphe relationnel...")
        hypergraph = self.builder.build_from_tables(
            tables, relationships, 
            temporal_columns=metrics.repeated_measurements
        )
        
        # Étape 3: Préparation des features
        print("Étape 3: Préparation des features...")
        features = self._prepare_features(hypergraph, tables, metrics)
        
        # Étape 4: Initialisation du modèle
        print("Étape 4: Initialisation du modèle...")
        self.model = self._initialize_model(recommendations)
        
        # Étape 5: Entraînement/Inférence
        print("Étape 5: Exécution...")
        if target_task:
            results = self._execute_task(hypergraph, features, target_task)
        else:
            results = self._generate_insights(hypergraph, features)
        
        # Étape 6: Évaluation
        print("Étape 6: Évaluation...")
        evaluation = self._evaluate_results(results, metrics)
        
        self.results = {
            'metrics': metrics,
            'recommendations': recommendations,
            'hypergraph': hypergraph,
            'features': features,
            'results': results,
            'evaluation': evaluation
        }
        
        print("=== MT5D Pipeline - Terminé ===")
        return self.results
    
    def save_pipeline(self, output_dir: str):
        """Sauvegarde le pipeline et les résultats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sauvegarder la configuration
        with open(output_path / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        # Sauvegarder les résultats
        with open(output_path / 'results.pkl', 'wb') as f:
            import pickle
            pickle.dump(self.results, f)
        
        # Sauvegarder le modèle si entraîné
        if self.model is not None:
            torch.save(self.model.state_dict(), output_path / 'model.pt')
        
        print(f"Pipeline sauvegardé dans: {output_dir}")
