import logging
from typing import Dict, List
from .mt5d_pipeline import MT5DPipeline

logger = logging.getLogger(__name__)

class OperationalOrchestrator:
    """
    Couche 4: Operational Orchestration Layer.
    Gère le cycle de vie de multiples pipelines d'analyse (ex: par département).
    """
    
    def __init__(self):
        self.pipelines: Dict[str, MT5DPipeline] = {}
        self.status: Dict[str, str] = {}
        
    def register_pipeline(self, name: str, config_path: str = None):
        """Enregistre une nouvelle configuration de pipeline."""
        logger.info(f"Enregistrement du pipeline : {name}")
        self.pipelines[name] = MT5DPipeline(config_path)
        self.status[name] = "idle"
        
    def run_all(self, data_registry: Dict[str, Dict]):
        """
        Exécute tous les pipelines enregistrés séquentiellement ou en parallèle.
        data_registry: { 'pipeline_name': {'tables': ..., 'rels': ...} }
        """
        results_agg = {}
        
        for name, pipeline in self.pipelines.items():
            if name not in data_registry:
                logger.warning(f"Pas de données pour le pipeline {name}, ignoré.")
                continue
                
            logger.info(f"Démarrage orchestration : {name}")
            self.status[name] = "running"
            
            try:
                data = data_registry[name]
                res = pipeline.run(data['tables'], data['rels'])
                results_agg[name] = res
                self.status[name] = "completed"
            except Exception as e:
                logger.error(f"Echec pipeline {name}: {e}")
                self.status[name] = "failed"
                
        return results_agg

    def get_dashboard_status(self):
        """Retourne l'état du système pour le monitoring."""
        return self.status
