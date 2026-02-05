from setuptools import setup, find_packages

setup(
    name="mt5d-framework",
    version="0.1.0-alpha",
    description="Multi-Table 5D Analysis Framework",
    author="MT5D Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "dgl>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "hydra-core>=1.3.0",
        "wandb>=0.15.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "benchmarks": ["hyperopt", "ray[tune]"],
    },
    python_requires=">=3.9",
)
