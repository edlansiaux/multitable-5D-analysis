"""Setup for multitable-5d-analysis."""

from setuptools import setup, find_packages

setup(
    name="multitable-5d-analysis",
    version="0.1.0",
    description=(
        "Relational Hypergraph Transformer: A Unified Framework "
        "for 5D Multi-Table Analysis"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Edouard Lansiaux, Hugo Kazzi, Aurélien Loison, Slim Hammadi, Emmanuel Chazard",
    author_email="edouard1.lansiaux@chu-lille.fr",
    url="https://github.com/edlansiaux/multitable-5d-analysis",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "dgl>=1.1",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "scipy>=1.11",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "ruff", "mypy"],
        "examples": ["matplotlib>=3.7"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
