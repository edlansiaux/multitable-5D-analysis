import os
from pathlib import Path

# Chemin absolu vers le dossier configs
CONFIG_DIR = Path(__file__).parent

# Chemins directs vers les fichiers de configuration
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"
MEDICAL_CONFIG_PATH = CONFIG_DIR / "medical_config.yaml"
FINANCIAL_CONFIG_PATH = CONFIG_DIR / "financial_config.yaml"

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "MEDICAL_CONFIG_PATH",
    "FINANCIAL_CONFIG_PATH"
]
