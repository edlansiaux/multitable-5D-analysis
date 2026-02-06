"""
MT5D Framework: Multi-Table 5D Analysis Framework
"""
__version__ = "0.1.0-alpha"
__author__ = "MT5D Research Team"
__license__ = "MIT"

# Exposition des composants principaux à la racine pour un accès direct
from .core.pipeline.mt5d_pipeline import MT5DPipeline
from .core.pipeline.orchestrator import OperationalOrchestrator
from .models.architectures.rht import RelationalHypergraphTransformer
from .models.embeddings.pente import PentE
from .datasets.synthetic import SyntheticMultiTableGenerator
from .datasets.mimic import MIMICDataset

# Liste des exports publics lors d'un 'from mt5d import *'
__all__ = [
    "MT5DPipeline",
    "OperationalOrchestrator",
    "RelationalHypergraphTransformer",
    "PentE",
    "SyntheticMultiTableGenerator",
    "MIMICDataset"
]
