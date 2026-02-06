# Contributing to MT5D Analysis Framework

First off, thank you for considering contributing to the **MT5D Analysis Framework**! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

Whether you're fixing a bug, adding a new dataset loader, improving the RHT architecture, or just fixing a typo in the documentation, your help is welcome.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Setting up the Environment](#setting-up-the-environment)
   - [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
   - [Branching Strategy](#branching-strategy)
   - [Coding Standards](#coding-standards)
   - [Testing](#testing)
4. [How to Contribute](#how-to-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
   - [Submitting Pull Requests](#submitting-pull-requests)
5. [Extending the Framework](#extending-the-framework)

---

## Code of Conduct

This project adheres to a standard Code of Conduct. We expect all contributors and participants to adhere to it in all interactions (issues, pull requests, discussions). Please be respectful, inclusive, and professional.

---

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher.
- **Git**: For version control.
- **Docker** (Optional): If you prefer developing in a containerized environment.

### Setting up the Environment

We use a `Makefile` to simplify common development tasks.

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/multitable-5d-analysis.git](https://github.com/YOUR-USERNAME/multitable-5d-analysis.git)
   cd multitable-5d-analysis
    ```
3. **Install dependencies** (Editable mode):
   ```bash
   make install
   ```
   **Note on DGL/PyTorch**: If you are using a specific hardware configuration (e.g., CUDA 11.x), the standard `pip install` might not pull the optimized versions. Please refer to the [DGL Getting Started](https://www.dgl.ai/pages/start.html) guide to install the correct version for your system if `make install` fails or defaults to CPU.
4. **Verify the installation** by running the test suite:
   ```bash
   make test
   ```
**Project Structure**
Before coding, please familiarize yourself with the directory structure:
- `mt5d/core/`: The engine room. Contains the `Pipeline`, `HypergraphBuilder`, and `Profiler`.
- `mt5d/models/`: Deep Learning architectures (`PentE` embeddings, `RHT` model).
- `mt5d/datasets/`: Data loaders. New datasets must inherit from `BaseDataset`.
- `configs/`: YAML configuration files used by the pipeline.
- `tests/`: Unit and integration tests using `pytest`.
- `examples/`: Standalone scripts demonstrating usage.

## Development Workflow

### Branching Strategy

1.  **Sync your fork** with the main repository (`upstream/main`).
2.  **Create a new branch** for your work. Use descriptive names:
    * `feature/add-finance-loader`
    * `fix/hypergraph-memory-leak`
    * `docs/improve-readme`
    
    ```bash
    git checkout -b feature/my-awesome-feature
    ```

### Coding Standards

We enforce high-quality code standards to maintain readability and reliability.

* **Formatter**: We use `black`.
* **Linter**: We use `flake8`.
* **Type Hints**: Python type hints are strongly encouraged.

Before committing, run the following commands to ensure your code complies:

```bash
# Auto-format your code
make format

# Check for linting errors
make lint
```

### Testing

We use **pytest**. All new features must include unit tests, and bug fixes must include regression tests.

* Run all tests:
    ```bash
    make test
    ```
* Run a specific test file:
    ```bash
    pytest tests/test_rht.py
    ```

---

## How to Contribute

### Reporting Bugs

If you find a bug, please check the [Issue Tracker](https://github.com/edlansiaux/multitable-5d-analysis/issues) to see if it has already been reported. If not, open a new issue using the **Bug Report** template.

**Please include:**
* OS, Python version, PyTorch/DGL versions.
* A minimal reproducible example (code snippet).
* Expected vs. actual behavior.

### Suggesting Enhancements

Have an idea for the framework? Open an issue using the **Feature Request** template.
* Describe the problem you are solving.
* Proposed solution or API change.
* Discuss it in the issue before spending a lot of time coding!

### Submitting Pull Requests

1.  Push your branch to your fork.
2.  Open a Pull Request against the `main` branch of the upstream repository.
3.  Fill out the PR template checklist:
    * [ ] Tests added/passed.
    * [ ] Documentation updated.
    * [ ] Linter/Formatter run.
4.  Wait for review. Be open to feedback and ready to make changes.

---

## Extending the Framework

Here are common ways to contribute to the core logic:

### 1. Adding a New Dataset
Create a new file in `mt5d/datasets/` (e.g., `retail.py`).
1.  Inherit from `mt5d.datasets.base.BaseDataset`.
2.  Implement `load()` to return the dictionary of DataFrames and the relationship list.
3.  Implement `get_temporal_columns()`.
4.  Expose it in `mt5d/datasets/__init__.py`.

### 2. Improving the 5D Embeddings
If you want to add a new dimension to **PentE** (e.g., Spatial):
1.  Modify `mt5d/models/embeddings/pente.py`.
2.  Add a new sub-encoder (e.g., `SpatialEncoder`).
3.  Update the `forward` pass to include the new dimension in the fusion mechanism.
4.  Update the `default_config.yaml` to include parameters for this dimension.

---

**Happy Coding!** 🚀
