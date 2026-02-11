# Multitable 5D Analysis — Relational Hypergraph Transformer (RHT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Position & Design Paper Implementation**
> Methodological and Conceptual Framework for 5D Multi-Table Analysis:
> A Unified Approach for Complex Data Reuse

## Overview

This repository provides the reference implementation of the **Relational Hypergraph Transformer (RHT)** architecture described in:

> Lansiaux E., Kazzi H., Loison A., Hammadi S., Chazard E. (2026).
> *Methodological and Conceptual Framework for 5D Multi-Table Analysis:
> A Unified Approach for Complex Data Reuse.*

The framework addresses **five critical dimensions** of complexity in multi-table data:

| # | Dimension | Challenge |
|---|-----------|-----------|
| D1 | **Massive Volume** | Scalability to millions of records across dozens of tables |
| D2 | **Multiplicity of Variables** | Hundreds of columns per table |
| D3 | **High Categorical Cardinality** | Thousands of distinct categories (e.g. ICD-10 codes) |
| D4 | **Multiple Tables & Relationships** | Complex relational schemas (1:N, N:M, hierarchical) |
| D5 | **Repeated Temporal Measurements** | Irregular time series across tables |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Relational Hypergraph Transformer         │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Module 1    │  │  Module 2    │  │   Module 3      │  │
│  │  Hypergraph  │→ │  Temporal    │→ │   High-Card.    │  │
│  │  Construction│  │  Embeddings  │  │   Attention     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│         │                                     │          │
│         └──────────────┐  ┌───────────────────┘          │
│                        ▼  ▼                              │
│               ┌─────────────────┐                        │
│               │   Module 4      │                        │
│               │   Relational    │                        │
│               │   Discovery     │                        │
│               └─────────────────┘                        │
│                        │                                 │
│                        ▼                                 │
│               ┌─────────────────┐                        │
│               │  PentE Embedding│                        │
│               │  (5D unified)   │                        │
│               └─────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Eight-Step Pipeline

1. **Predictive Meta-Profiling** — Automatic schema characterization
2. **Automated Hypergraph Construction** — Relational schema → hypergraph
3. **Unified 5D Embedding (PentE)** — Pentadimensional latent space
4. **Relational Contrastive Learning** — Contrastive training on relational structure
5. **Dynamic Graph Rewiring** — Task-adaptive relational graph reweighting
6. **Relational Causal Inference** — Cross-table causal discovery
7. **Federated Multi-Table Learning** — Privacy-preserving distributed training
8. **Operationalization & Monitoring** — Deployment, drift detection, explainability

## Installation

```bash
git clone https://github.com/edlansiaux/multitable-5d-analysis.git
cd multitable-5d-analysis
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1
- DGL >= 1.1 (Deep Graph Library)
- NumPy, Pandas, Scikit-learn

## Quick Start

```python
from rht import RelationalHypergraphTransformer
from pipeline import MultiTablePipeline

# Load your relational tables
tables = {
    "patients": patients_df,
    "admissions": admissions_df,
    "diagnoses": diagnoses_df,
    "prescriptions": prescriptions_df,
    "lab_results": lab_results_df,
}

# Define schema relationships
relations = [
    ("patients", "patient_id", "admissions", "patient_id"),
    ("admissions", "hadm_id", "diagnoses", "hadm_id"),
    ("admissions", "hadm_id", "prescriptions", "hadm_id"),
    ("admissions", "hadm_id", "lab_results", "hadm_id"),
]

# Run the full 8-step pipeline
pipeline = MultiTablePipeline(tables, relations)
pipeline.run()

# Access PentE embeddings
embeddings = pipeline.get_pente_embeddings()

# Predict (e.g. hospital mortality)
predictions = pipeline.predict(task="mortality")
```

## Repository Structure

```
multitable-5d-analysis/
├── rht/                          # Core RHT architecture
│   ├── __init__.py
│   ├── model.py                  # Main RelationalHypergraphTransformer
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── hypergraph.py         # Module 1: Hypergraph Construction
│   │   ├── temporal.py           # Module 2: Multi-Scale Temporal Embeddings
│   │   ├── attention.py          # Module 3: High-Cardinality Attention
│   │   └── discovery.py          # Module 4: Differentiable Relational Discovery
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── pente.py              # PentE: Pentadimensional Embedding
│   │   ├── semantic.py           # Semantic encoder
│   │   ├── categorical.py        # Hierarchical categorical encoder
│   │   └── volume.py             # Volume normalizer
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── contrastive.py        # Relational-Temporal Contrastive Loss
│   │   └── relational.py         # Relational Discovery Loss
│   └── utils/
│       ├── __init__.py
│       ├── schema.py             # Schema detection utilities
│       └── temporal.py           # Temporal utilities
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py               # 8-step MultiTablePipeline
│   ├── profiling.py              # Step 1: Meta-Profiling
│   ├── rewiring.py               # Step 5: Dynamic Graph Rewiring
│   ├── causal.py                 # Step 6: Relational Causal Inference
│   └── federated.py              # Step 7: Federated Multi-Table Learning
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # 5D evaluation metrics (Section 7.2)
├── configs/
│   └── default.yaml              # Default hyperparameters
├── tests/
│   ├── __init__.py
│   └── test_modules.py           # Unit tests
├── examples/
│   └── mimic_iv_example.py       # MIMIC-IV walkthrough
├── setup.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Benchmarks (MT-5D-Bench)

Planned benchmark datasets (Section 7.1):

| Dataset | Tables | Records | Cardinality | Temporal |
|---------|--------|---------|-------------|----------|
| MIMIC-IV | 26 | 15M+ measurements | 10k+ ICD codes | Irregular vitals |
| Amazon Multi-Table | 5 | Millions | Hierarchical categories | Interaction TS |
| Financial Transactions | Multiple | High-frequency | Transaction codes | Millisecond |

Benchmark suite will be hosted at: [mt5d-benchmark.org](https://mt5d-benchmark.org)

## Anticipated Results (Design Targets)

| Metric | SOTA (TGNN) | RHT (target) | Improvement |
|--------|-------------|---------------|-------------|
| Rare Category F1 | 0.45 | 0.78 | +73% |
| Relation Discovery Precision | 0.67 | 0.92 | +37% |
| Irregular TS Forecasting RMSE | 1.23 | 0.89 | -28% |
| Training Time (100 tables) | 48h | 12h | -75% |
| Memory Usage | 64 GB | 24 GB | -62% |
| Overall 5D Score | 0.58 | 0.83 | +43% |

> **Warning**: These are **research targets**, not confirmed empirical results.
> See Section 7.4 of the paper for the validation plan.

## Citation

```bibtex
@article{lansiaux2026multitable5d,
  title={Methodological and Conceptual Framework for 5D Multi-Table Analysis:
         A Unified Approach for Complex Data Reuse},
  author={Lansiaux, Edouard and Kazzi, Hugo and Loison, Aur{\'e}lien
          and Hammadi, Slim and Chazard, Emmanuel},
  year={2026},
  institution={Lille University Hospital, Lille Centrale Institute}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Authors

- **Edouard Lansiaux** — Emergency Department, Lille University Hospital
- **Hugo Kazzi** — Lille Centrale Institute
- **Aurelien Loison** — Lille Centrale Institute
- **Pr. Slim Hammadi** — Lille Centrale Institute, CRISTAL UMR CNRS 9189
- **Pr. Emmanuel Chazard** — Department of Public Health, EA 2694, ULR 2694-METRICS
