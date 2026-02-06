"""
Test suite for the MT5D framework.
"""

# Initialisation du package de tests pour MT5D
# Expose les modules de tests pour faciliter l'introspection et l'exécution programmatique

from . import conftest
from . import test_pipeline
from . import test_rht
from . import test_pente
from . import test_hypergraph
from . import test_profiling
from . import test_evaluation
from . import test_utils

__all__ = [
    "conftest",
    "test_pipeline",
    "test_rht",
    "test_pente",
    "test_hypergraph",
    "test_profiling",
    "test_evaluation",
    "test_utils"
]
