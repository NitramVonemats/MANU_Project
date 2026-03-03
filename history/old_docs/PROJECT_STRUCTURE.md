# MANU Project - Directory Structure

**Last Updated:** 2026-01-19
**Status:** Clean, Professional, Publication-Ready

---

## 📁 Directory Structure

```
MANU_Project/
│
├── README.md                     # ⭐ Main project README (START HERE!)
├── PROJECT_STRUCTURE.md          # This file - Directory structure guide
│
├── src/                          # Main Python code
│   ├── core/
│   │   ├── optimized_gnn.py     # GNN training pipeline
│   │   └── model_comparison.py  # Model comparison utilities
│   └── utils/
│       └── check_progress.py    # Progress tracking
│
├── adme_gnn/                     # Core package
│   ├── data/                    # Data loading & graph construction
│   ├── models/                  # GNN architectures & foundation models
│   └── utils/                   # Utility functions
│
├── optimization/                 # HPO algorithms (6 algorithms)
│   ├── random_search.py
│   ├── pso_optimizer.py         # Particle Swarm Optimization
│   ├── abc_optimizer.py         # Artificial Bee Colony
│   ├── genetic_optimizer.py     # Genetic Algorithm
│   ├── simulated_annealing.py   # Simulated Annealing
│   └── hill_climbing.py         # Hill Climbing
│
├── scripts/                      # Analysis & benchmarking scripts
│   ├── run_hpo.py               # ⭐ Main HPO runner
│   ├── create_hpo_visualizations.py
│   ├── benchmark_report.py
│   └── analyses/
│       ├── benchmark_foundation_models.py
│       └── create_visualizations.py
│
├── config/                       # Configuration files
│   └── benchmarking/
│       ├── config_benchmark.yaml
│       └── config_foundation_benchmark.yaml
│
├── results/                      # ⭐ All experimental results
│   ├── hpo/                     # HPO results (36 JSON files)
│   │   ├── Caco2_Wang/          # 6 algorithms × JSON
│   │   ├── Half_Life_Obach/
│   │   ├── Clearance_Hepatocyte_AZ/
│   │   ├── Clearance_Microsome_AZ/
│   │   ├── tox21/
│   │   └── herg/
│   ├── foundation_models/       # Foundation model results
│   └── benchmark_*/             # Benchmark reports
│
├── figures/                      # ⭐ All visualizations (70+ plots)
│   ├── comparative/             # Unified plots (5 PNG)
│   ├── hpo/                     # HPO results (4 PNG)
│   ├── ablation_studies/        # Ablation studies (12 PNG)
│   └── per_dataset_analysis/    # Per-dataset (42 PNG)
│
├── docs/                         # Documentation
│   └── STATUS/                  # ⭐ Project status files
│       ├── FINAL_PROJECT_STATUS.md      # Complete summary (READ THIS!)
│       ├── EXECUTION_STATUS.md          # Technical details
│       ├── FOUNDATION_MODELS_STATUS.md  # SOTA comparison
│       └── README.md                    # Status docs index
│
├── logs/                         # All log files
│   ├── hpo_toxicity.log
│   ├── hpo_toxicity_full.log
│   └── foundation_benchmark.log
│
├── data/                         # Raw data & TDC cache
├── datasets/                     # Dataset metadata
├── archive/                      # Old files
│
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
├── setup.py                      # Package setup
└── .gitignore                    # Git ignore rules
```

---

## 🎯 Key Directories

### **`README.md`** - Start Here!
Professional README with:
- Project overview
- Results summary
- Quick start guide
- Installation instructions
- Publication targets

### **`docs/STATUS/`** - Project Status
Complete project documentation:
- **`FINAL_PROJECT_STATUS.md`** - ⭐ **READ THIS FIRST** - Complete project summary
- **`EXECUTION_STATUS.md`** - Technical execution details
- **`FOUNDATION_MODELS_STATUS.md`** - GNN vs SOTA comparison

### **`results/`** - All Experimental Results
Centralized location for all outputs:
- **`hpo/`** - 36 HPO JSON files (6 datasets × 6 algorithms)
- **`foundation_models/`** - Foundation model benchmark CSV
- **`benchmark_*/`** - Comprehensive reports

### **`figures/`** - Publication-Ready Visualizations
70+ high-quality plots:
- **`comparative/`** - Cross-dataset comparisons (5 plots)
- **`hpo/`** - HPO algorithm analysis (4 plots)
- **`ablation_studies/`** - Hyperparameter analysis (12 plots)
- **`per_dataset_analysis/`** - Individual dataset details (42 plots)

### **`src/`** - Main Python Code
Organized by purpose:
- **`core/`** - GNN training (`optimized_gnn.py`), model comparison
- **`utils/`** - Utility scripts

### **`config/benchmarking/`** - Configuration
All YAML configuration files for experiments

### **`logs/`** - Execution Logs
All `.log` and `.pid` files

---

## 📊 File Counts

| Category | Count | Location |
|----------|-------|----------|
| **HPO Results** | 36 JSON | `results/hpo/*/` |
| **Visualizations** | 70+ PNG | `figures/*/` |
| **Datasets** | 6 | TDC (4 ADME + 2 Tox) |
| **Status Docs** | 5 MD | `docs/STATUS/` |
| **Config Files** | 2 YAML | `config/benchmarking/` |
| **Python Modules** | 3 | `src/core/`, `src/utils/` |

---

## 🚀 Quick Navigation

### **To Run Experiments:**
```bash
python scripts/run_hpo.py --dataset Caco2_Wang --algo pso --trials 10
```

### **To Generate Visualizations:**
```bash
python scripts/create_hpo_visualizations.py
```

### **To View Results:**
- HPO: `results/hpo/*/hpo_*.json`
- Foundation: `results/foundation_models/`
- Plots: `figures/*/`

### **To Read Documentation:**
1. `README.md` - Overview & quick start
2. `docs/STATUS/FINAL_PROJECT_STATUS.md` - Complete summary
3. `docs/STATUS/EXECUTION_STATUS.md` - Technical details

---

## ✅ Organization Principles

### **Clean Root Directory**
- Only essential files in root (README, requirements, config)
- No clutter (logs, status files moved to subdirs)

### **Logical Grouping**
- **Source code** → `src/`
- **Results** → `results/`
- **Docs** → `docs/`
- **Logs** → `logs/`
- **Config** → `config/`

### **Publication-Ready**
- Professional structure
- Clear documentation
- Easy to navigate
- Reproducible

---

## 📝 Documentation Hierarchy

1. **`README.md`** - Start here (overview, results, quick start)
2. **`PROJECT_STRUCTURE.md`** - This file (directory guide)
3. **`docs/STATUS/FINAL_PROJECT_STATUS.md`** - Complete project summary
4. **`docs/STATUS/EXECUTION_STATUS.md`** - Technical execution details

---

*Last Updated: 2026-01-19*
*Status: Clean, Professional, Publication-Ready* ✅
