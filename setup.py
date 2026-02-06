from setuptools import setup, find_packages

setup(
    name="multitable-5d-analysis",
    version="0.1.0",
    description="Implementation of the 5D Multi-Table Analysis Framework (RHT)",
    author="Edouard Lansiaux",
    author_email="edouard1.lansiaux@chu-lille.fr",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "dgl>=0.9.0",  # Deep Graph Library
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pyyaml>=5.4.0",
        "networkx>=2.6.0"
    ],
    python_requires=">=3.8",
)
