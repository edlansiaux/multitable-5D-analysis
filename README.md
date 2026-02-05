# MT5D Framework: Multi-Table 5D Analysis

## Overview
MT5D is an open-source framework for analyzing complex multi-table datasets across 5 dimensions:
1. **Volume**: Large-scale data handling
2. **Many Variables**: High-dimensional feature spaces
3. **High Cardinality**: Categorical variables with many unique values
4. **Many Tables & Relationships**: Complex relational schemas
5. **Repeated Measurements**: Temporal and longitudinal data

## Installation

```bash
# Install from source
git clone https://github.com/your-org/mt5d-framework.git
cd mt5d-framework
pip install -e .

# Or install with specific components
pip install -e .[dev,docs,benchmarks]
```

## Quick start
```bash
from mt5d import MT5DPipeline
import pandas as pd

# Load your multi-table data
tables = {
    'table1': pd.read_csv('data/table1.csv'),
    'table2': pd.read_csv('data/table2.csv'),
}

# Define relationships between tables
relationships = [
    ('table1', 'id', 'table2', 'foreign_id', 'relation_type')
]

# Initialize and run the pipeline
pipeline = MT5DPipeline()
results = pipeline.run(tables, relationships)

# Access results
print(f"5D Metrics: {results['metrics']}")
print(f"Generated Insights: {results['results']['insights']}")
```

## Key Features
### 1. Automated 5D Profiling
```bash
from mt5d.core.profiling import DimensionalProfiler

profiler = DimensionalProfiler()
metrics = profiler.profile(tables, relationships)
recommendations = profiler.recommend_pipeline()
```
### 2. Relational Hypergraph Construction
```bash
from mt5d.core.hypergraph import RelationalHypergraphBuilder

builder = RelationalHypergraphBuilder()
hypergraph = builder.build_from_tables(tables, relationships)
```

### 3. Pentadimensional Embeddings (PentE)
```bash
from mt5d.models.embeddings import PentE

pente = PentE()
embeddings = pente(node_features, relation_features, ...)
```

### 4. Relational Hypergraph Transformer
```bash
from mt5d.models.architectures import RelationalHypergraphTransformer

model = RelationalHypergraphTransformer(
    input_dim=128,
    hidden_dim=256,
    output_dim=64
)
output = model(hypergraph, node_features)
```

## Documentation
Full documentation available at: docs.mt5d.ai
- API Reference
- Tutorials
- Benchmarks

## Contributing
Please read our Contributing Guidelines and Code of Conduct.

 ## License
MIT License - see LICENSE for details.
### **Deliverables in 6 Months**

1. **Public GitHub repository** with complete source code
2. **Documentation techniques** and tutorials
3. **Initial PyPI package** (alpha version)
4. **Docker image** for reproducible environment
5. **Basic benchmarks** on public datasets
6. **Use cases** for 3 domains
7. **Unit tests** with >80% coverage
8. **Operational CI/CD pipeline**
9. **Academic preprint** on arXiv
10. **Initial community** (Discord/Slack, initial contributions)

### **Immediate Next Steps**

1. **Week 13**: Finalize tests and documentation
2. **Week 14**: Prepare for open-source launch
3. **Weeks 15-16**: Collect community feedback
4. **Weeks 17-18:** Implement priority improvements
5. **Weeks 19-20:** Prepare the MT-5D Benchmark
6. **Weeks 21-24:** Optimize performance and scalability

This implementation provides a solid foundation for the open-source framework, ready to be extended and improved by the community.
