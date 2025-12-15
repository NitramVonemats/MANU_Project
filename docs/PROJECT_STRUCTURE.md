# Project Structure

Detailed directory layout of the MANU project.

---

## Directory Layout

```
MANU_Project/
├── adme_gnn/                    # Main Python package
│   ├── __init__.py
│   ├── models/                  # Neural network architectures
│   │   ├── __init__.py
│   │   ├── gnn.py              # GNN backbones (Graph, GCN, GAT, etc.)
│   │   ├── foundation.py       # Foundation model encoders
│   │   └── predictors.py       # Prediction heads (GNN-only, hybrid, etc.)
│   ├── data/                    # Data handling
│   │   ├── __init__.py
│   │   ├── featurizer.py       # Molecular featurization
│   │   └── loader.py           # Dataset loaders and graph construction
│   ├── training/                # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training and evaluation loops
│   │   └── benchmark.py        # Benchmarking utilities
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── transforms.py       # Data transformations
│   │   └── utils.py            # General utilities
│   ├── configs/                 # Configuration management
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   └── model_config.py
│   ├── best_models/             # Best model configurations
│   └── visualizations/          # Visualization utilities
│
├── optimization/                # Hyperparameter optimization
│   ├── __init__.py
│   ├── problem.py              # Optimization problem definition
│   ├── space.py                # Hyperparameter search space
│   ├── runner.py               # HPO runner
│   └── algorithms/             # Optimization algorithms
│       ├── __init__.py
│       ├── random_search.py    # Random search
│       ├── genetic.py          # Genetic algorithm
│       ├── pso.py              # Particle swarm optimization
│       ├── simulated_annealing.py
│       ├── hill_climbing.py
│       └── abc.py              # Artificial bee colony
│
├── scripts/                     # Utility scripts
│   ├── analyses/                # Analysis scripts
│   │   ├── tanimoto_similarity_analysis.py
│   │   ├── label_distribution_analysis.py
│   │   ├── feature_label_correlation_analysis.py
│   │   ├── run_all_analyses.py
│   │   └── benchmark_foundation_models.py
│   ├── visualization/           # Visualization scripts
│   │   ├── create_publication_figures.py
│   │   └── visualize_results.py
│   ├── run_hpo.py              # Main HPO runner
│   ├── benchmark_report.py     # Benchmark reporting
│   └── generate_report.py      # Report generation
│
├── experiments/                 # Experiment outputs
│   ├── runs/                    # HPO run results (JSON files)
│   ├── reports/                 # Benchmark reports
│   └── checkpoints/             # Model checkpoints
│
├── figures/                     # All visualizations
│   ├── comparative/             # Unified plots (all datasets)
│   ├── hpo/                    # HPO results visualizations
│   ├── ablation_studies/       # Hyperparameter analysis
│   └── per_dataset_analysis/   # Per-dataset detailed analysis
│       ├── Caco2_Wang/
│       ├── Clearance_Hepatocyte_AZ/
│       ├── Clearance_Microsome_AZ/
│       ├── Half_Life_Obach/
│       ├── Tox21/
│       ├── hERG/
│       └── ClinTox/
│
├── runs/                        # HPO run results
│   ├── Caco2_Wang/
│   ├── Clearance_Hepatocyte_AZ/
│   ├── Clearance_Microsome_AZ/
│   └── Half_Life_Obach/
│
├── docs/                        # Documentation
│   ├── PROJECT_STRUCTURE.md    # This file
│   ├── GUIDES.md               # Usage guides
│   ├── DATASETS.md             # Dataset documentation (to be created)
│   └── METHODS.md              # Methods documentation (to be created)
│
├── archive/                     # Archived/deprecated code
│   ├── summaries/              # Old summary/log MD files
│   ├── old_results/            # Old CSV/PNG files
│   ├── old_src/                # Previous src/ folder
│   └── ...                     # Other archived content
│
├── tests/                       # Unit tests (to be implemented)
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
│
├── data/                        # Downloaded datasets (TDC cache)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project README
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── pyproject.toml              # Modern Python packaging
├── config_benchmark.yaml       # Benchmark configuration
└── config_benchmark.json       # Alternative config format

```

## Key Changes from Previous Structure

### Renamed Directories
- `GNN_test/` → `adme_gnn/` (more professional, descriptive name)
- `GNN_test/graph/` → `adme_gnn/data/` (clearer purpose)
- `GNN_test/functional/` → `adme_gnn/utils/` (standard naming)
- `GNN_test/services/` → `adme_gnn/training/` (clearer purpose)
- `hpo/` → `optimization/` (more descriptive)

### Reorganized Structure
- Consolidated all experiment outputs into `experiments/`
- Moved all visualizations to `outputs/figures/`
- Organized scripts into `analyses/` and `visualization/` subdirectories
- Created proper `tests/` directory
- Added proper package files (`setup.py`, `pyproject.toml`)

### Removed Clutter
- Archived old CSV/PNG files
- Removed duplicate `MANU_Project/` folder
- Cleaned up root directory
- Moved legacy code to `archive/`

## Import Structure

### Within the Package
```python
# Models
from adme_gnn.models.gnn import GNNBackbone
from adme_gnn.models.foundation import ChemBERTaEncoder
from adme_gnn.models.predictors import GNNOnlyRegressor

# Data
from adme_gnn.data.featurizer import enhanced_atom_features
from adme_gnn.data.loader import build_loaders

# Training
from adme_gnn.training.trainer import train_model
from adme_gnn.training.benchmark import benchmark_dataset

# Utils
from adme_gnn.utils.metrics import compute_metrics_np
from adme_gnn.utils.transforms import transform_y
```

### From Scripts
```python
# Always use absolute imports from adme_gnn
from adme_gnn.models.foundation import ChemBERTaEncoder
from adme_gnn.data.loader import build_loaders
```

## Running the Project

### Installation
```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Training
```bash
# Run HPO for all datasets
python scripts/run_hpo.py --config config_benchmark.yaml

# Run specific dataset
python scripts/run_hpo.py --datasets Half_Life_Obach
```

### Analysis
```bash
# Run all analyses
python scripts/analyses/run_all_analyses.py

# Run specific analysis
python scripts/analyses/tanimoto_similarity_analysis.py
```

### Benchmarking
```bash
# Generate benchmark report
python scripts/benchmark_report.py --config config_benchmark.json
```

## Version History

- **v3.0.0** - Complete project reorganization (November 2025)
- **v2.0.0** - Foundation model integration
- **v1.0.0** - Initial GNN implementation
