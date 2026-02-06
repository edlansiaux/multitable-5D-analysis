"""
API principale pour MT5D
"""
from typing import Dict, Any
import pandas as pd
from ..core.pipeline.mt5d_pipeline import MT5DPipeline

def analyze_multitable_data(
    tables: Dict[str, pd.DataFrame],
    relationships: list,
    config: Dict[str, Any] = None
):
    """
    Fonction helper pour lancer l'analyse complète
    """
    if config is None:
        config = {
            "hidden_dim": 256,
            "output_dim": 64,
            "epochs_pretrain": 5
        }
        
    pipeline = MT5DPipeline(config)
    model = pipeline.run(tables, relationships)
    
    return model, pipeline.metrics.generate_report()
