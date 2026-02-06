from .benchmark_runner import run_benchmark
from .download_example_data import main as download_data

# Si generate_docs.py a été créé
try:
    from .generate_docs import main as generate_docs
except ImportError:
    pass

__all__ = ["run_benchmark", "download_data", "generate_docs"]
