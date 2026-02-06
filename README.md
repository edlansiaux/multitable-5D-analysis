# MT5D: Relational Hypergraph Transformer Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](docker/Dockerfile)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](.github/workflows/ci.yml)

**Official implementation of the paper:** *"Methodological and Conceptual Framework for 5D Multi-Table Analysis: A Unified Approach for Complex Data Reuse"*.

## 📖 Overview

**MT5D** is an open-source framework designed to solve the complexity of modern relational data. [cite_start]Unlike traditional approaches that flatten tables (joins) or treat them as simple graphs, MT5D introduces a **Relational Hypergraph Transformer (RHT)** to simultaneously address the **5 Dimensions of Complexity**[cite: 9, 35, 36]:

1.  **📊 Volume**: Scalable handling of massive datasets via sparse attention.
2.  **🧩 Many Variables**: Management of high-dimensional feature spaces.
3.  **🗂️ High Cardinality**: Hierarchical encoding for variables with thousands of unique values (e.g., ICD-10 codes).
4.  **🔗 Many Tables**: Adaptive Hypergraph construction for complex n-ary schemas.
5.  **⏱️ Repeated Measurements**: Temporal and longitudinal data integration via multi-scale embeddings.

## 🚀 Key Features

* [cite_start]**PentE (Pentadimensional Embeddings):** A unified embedding space integrating semantic, relational, temporal, categorical, and volumetric information[cite: 11, 221].
* [cite_start]**Relational Hypergraph Transformer (RHT):** A deep learning architecture replacing standard joins with learnable message passing on hypergraphs[cite: 10, 141].
* [cite_start]**Automated Profiling:** Auto-detection of schemas, foreign keys, and temporal columns[cite: 204].
* [cite_start]**Causal Inference:** Discovery of latent causal links between tables beyond simple correlation[cite: 247].
* [cite_start]**Explainability:** Extraction of strong relational patterns driving predictions[cite: 260].
* **Drift Detection:** Monitoring of relational distribution shifts in production.

## 🛠️ Installation

### From Source

```bash
git clone [https://github.com/edlansiaux/multitable-5d-analysis.git](https://github.com/edlansiaux/multitable-5d-analysis.git)
cd multitable-5d-analysis

# Standard installation
pip install -e .
```
### Using Makefile (Recommended)
```bash
# Install dependencies
make install

# Run unit tests
make test
```

### Using Docker
```bash
# Build the API image
make docker-build

# Run the inference server
make docker-run
```

## ⚡ Quick Start/
### 1. Medical Analysis Example (MIMIC-IV Style)

```bash
from mt5d.core.pipeline.mt5d_pipeline import MT5DPipeline
from mt5d.datasets.synthetic import SyntheticMultiTableGenerator

# 1. Generate synthetic medical data (Patients, Admissions, Diagnoses, LabEvents)
generator = SyntheticMultiTableGenerator(num_patients=500)
tables, relationships = generator.generate()

# 2. Initialize the 5D Pipeline
# Automatically handles Profiling -> Hypergraph -> Embedding -> RHT Training
pipeline = MT5DPipeline(config_path="configs/medical_config.yaml")

# 3. Run analysis for a specific task
results = pipeline.run(
    tables=tables, 
    relationships=relationships, 
    target_task="mortality_prediction"
)

# 4. Access Insights
metrics = results['evaluation']
print(f"Global 5D Score: {metrics['volume_score']:.4f}")
print(f"Rare Category Recall: {metrics.get('rare_category_recall', 0.0):.4f}")
```

### 2. Operational API Usage
Start the server:
```bash
python mt5d/api/main.py
```

Query the API:
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"task_name": "fraud_detection", "epochs": 10}'
```

## 🏗️ Architecture
```bash
multitable-5d-analysis/
├── mt5d/
│   ├── core/
│   │   ├── pipeline/       # Orchestrator & Main Pipeline
│   │   ├── hypergraph/     # Adaptive Hypergraph Builder
│   │   ├── profiling/      # Dimensional & Meta Profiler
│   │   ├── causal/         # Relational Causal Inference
│   │   └── explainability/ # Subgraph extraction
│   ├── models/
│   │   ├── architectures/  # Relational Hypergraph Transformer (RHT)
│   │   └── embeddings/     # PentE (5D Embeddings)
│   ├── datasets/           # Loaders (MIMIC, Amazon, Finance)
│   └── api/                # FastAPI Service
├── configs/                # YAML Configurations
├── docker/                 # Dockerfiles
├── examples/               # Domain-specific scripts
└── tests/                  # Pytest suite
```

## 📊 Benchmarks
MT5D aims to surpass state-of-the-art methods (TGN, GraphSAGE) on complex multi-table tasks

To reproduce these results, run:
```bash
python scripts/benchmark_runner.py --dataset synthetic_medical
```

## 🗺️ Roadmap & Status
```bash
[x] Core Framework: Pipeline, Profiler, Hypergraph Builder.
[x] Models: PentE and RHT implementation (PyTorch/DGL).
[x] Datasets: Loaders for MIMIC-IV, Amazon, and Finance.
[x] Infrastructure: Docker, CI/CD, and API.
[ ] Optimization: Distributed training for >1TB datasets.
[ ] Federated Learning: Full implementation of privacy-preserving cross-silo learning.
[ ] LLM Integration: Text-to-SQL-to-Hypergraph interface.
```
## 🤝 Contributing
We welcome contributions! Please see CONTRIBUTING.md for details on how to submit pull requests, report issues, or request features.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE]() file for details.

## 📄 Citation
If you use this code in your research, please cite the following paper:
```bash
@article{lansiaux2026mt5d,
  title={Methodological and Conceptual Framework for 5D Multi-Table Analysis: A Unified Approach for Complex Data Reuse},
  author={Lansiaux, Edouard},
  journal={Preprint},
  year={2026}
}
```
