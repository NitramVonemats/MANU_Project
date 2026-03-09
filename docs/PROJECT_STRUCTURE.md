# MANU Project Structure

## 📁 Directory Layout (Clean & Organized)

```
MANU/
│
├── 📄 MAIN PROJECT FILES
│   ├── README.md                    ← Start here!
│   ├── requirements.txt             ← Python dependencies
│   ├── reorganize_project.py        ← Structure reorganization script
│   ├── README_CLEAN.md             ← Reorganized readme
│   └── .gitignore
│
├── 📕 paper_1/                     ✅ FINAL PAPER
│   ├── main.tex                    ← LaTeX manuscript
│   ├── refs.bib                    ← Bibliography
│   └── images/                     ← Paper figures
│       ├── Data origin and Class distribution-2026-02-23-194726.png
│       ├── 01_algorithm_performance.png
│       ├── Benchmark-Architecture-GNN.png
│       ├── 05_classification_performance.png
│       ├── confusion_matrices.png
│       ├── foundation_ranking.png
│       ├── gnn_vs_foundation_comparison.png
│       ├── hpo_convergence_curves.png
│       └── param_sensitivity_heatmap.png
│
├── 📊 results/                     ✅ FINAL RESULTS ONLY
│   ├── hpo/                        ← HPO JSON files (37 total)
│   │   ├── Caco2_Wang/             ← 6 algorithm variants
│   │   │   ├── hpo_Caco2_Wang_random.json
│   │   │   ├── hpo_Caco2_Wang_pso.json
│   │   │   ├── hpo_Caco2_Wang_abc.json
│   │   │   ├── hpo_Caco2_Wang_ga.json
│   │   │   ├── hpo_Caco2_Wang_sa.json
│   │   │   └── hpo_Caco2_Wang_hc.json
│   │   ├── Half_Life_Obach/        ← 6 variants
│   │   ├── Clearance_Hepatocyte_AZ/ ← 6 variants
│   │   ├── Clearance_Microsome_AZ/  ← 6 variants
│   │   ├── herg/                    ← Classification
│   │   ├── tox21/                   ← Classification
│   │   └── foundation/              ← Foundation model baseline results
│   │
│   ├── figures/                    ← ONLY 9 publication-ready figures
│   │   ├── 01_algorithm_performance.png
│   │   ├── confusion_matrices.png
│   │   ├── foundation_ranking.png
│   │   ├── gnn_vs_foundation_comparison.png
│   │   ├── hpo_convergence_curves.png
│   │   ├── learning_curves.png
│   │   ├── multi_seed_boxplots.png
│   │   ├── param_sensitivity_heatmap.png
│   │   └── tpe_optimization_history.png
│   │
│   └── summary/                    ← Summary statistics
│       └── FINAL_RESULTS_SUMMARY.md
│
├── 📚 docs/                        ✅ CLEAN DOCUMENTATION
│   ├── README.md                   ← Doc overview
│   ├── METHODOLOGY.md              ← Experimental setup
│   ├── DATASETS.md                 ← Dataset descriptions
│   ├── PROJECT_STRUCTURE.md        ← This file
│   └── FORENSIC_ANALYSIS.md        ← Results authenticity report
│
├── 💻 code/                        ✅ SOURCE CODE
│   ├── models/
│   ├── utils/
│   ├── optimization/
│   └── evaluation/
│
├── 🔧 scripts/                     ✅ PRODUCTION SCRIPTS
│   ├── run_hpo_benchmark.py
│   ├── run_molclr_pretrained_benchmark.py
│   ├── run_chemberta_finetune.py
│   ├── run_multi_seed_validation.py
│   ├── run_tpe_benchmark.py
│   └── [other utility scripts]
│
└── 📦 archive/                     ✅ OLD/OUTDATED FILES (organized)
    ├── old_experiments/            ← All history/ + old runs
    │   ├── history/
    │   ├── GNN_test/
    │   └── [intermediate experiments]
    │
    ├── old_figures/                ← Intermediate/exploratory figures (57 files)
    │   ├── algorithm_ranking_heatmap.png
    │   ├── chemberta_overfitting_analysis.png
    │   ├── [41 more intermediate figures]
    │   └── ...
    │
    ├── old_reports/                ← Old benchmark reports
    │   └── [reports from earlier runs]
    │
    ├── old_scripts/                ← Unused/development scripts
    │   └── [scripts not in current use]
    │
    ├── old_documentation/          ← Outdated docs & Word files
    │   ├── MANU_MASSIVE_DOCUMENTATION_*.docx
    │   ├── DOCUMENTATION_COMPLETE.md
    │   ├── PROJECT_STRUCTURE.md
    │   └── [other old docs]
    │
    └── old_code/                   ← Old code implementations
        ├── adme_gnn/
        └── [deprecated code]
```

---

## 📍 What's Where

### For Reading the Paper
```
paper_1/
├── main.tex          ← Open in LaTeX editor
└── images/           ← Paper figures referenced in manuscript
```

### For Accessing Results
```
results/
├── hpo/              ← All 37 HPO result JSON files
│   └── [DATASET]/[ALGORITHM].json
├── figures/          ← 9 publication-ready figures
└── summary/          ← Summary statistics markdown
```

### For Understanding the Project
```
docs/
├── README.md         ← Project overview (start here)
├── METHODOLOGY.md    ← How experiments were done
├── DATASETS.md       ← Dataset descriptions
└── PROJECT_STRUCTURE.md ← This file
```

### For Running Experiments
```
scripts/
├── run_hpo_benchmark.py          ← Main HPO runner
├── run_multi_seed_validation.py  ← Multi-seed validation
└── [other scripts]
```

### For Archival
```
archive/
├── old_experiments/  ← All experiments prior to final run
├── old_figures/      ← 57 intermediate/exploratory figures
├── old_reports/      ← Earlier benchmark reports
├── old_scripts/      ← Unused scripts
└── old_documentation/ ← .docx files, draft docs
```

---

## 📊 File Counts

| Category | Count | Status |
|----------|-------|--------|
| **HPO Results** | 37 | ✅ Active |
| **Publication Figures** | 9 | ✅ Active |
| **Archived Figures** | 57 | 📦 Archive |
| **Documentation Files** | 4 | ✅ Active |
| **Archived Docs** | 5+ | 📦 Archive |
| **Scripts** | 5+ | ✅ Active |
| **Total Archived** | 100+ | 📦 Archive |

---

## 🔑 Key Files to Know

| File | Purpose | Location |
|------|---------|----------|
| **main.tex** | Final paper | `paper_1/` |
| **FINAL_RESULTS_SUMMARY.md** | Key findings | `results/summary/` |
| **README.md** | Start here | `docs/` |
| **METHODOLOGY.md** | How it was done | `docs/` |
| **DATASETS.md** | Dataset info | `docs/` |
| **hpo_[DATASET]_[ALGO].json** | Detailed results | `results/hpo/` |
| **01_algorithm_performance.png** | Main figure | `results/figures/` |

---

## 🗑️ What Was Archived

### From `history/`
- **Period:** Early development
- **Content:** 40+ intermediate experiments
- **Reason:** Superseded by current results
- **Location:** `archive/old_experiments/history/`

### From `figures/`
- **Total:** 57 old figures archived
- **Kept:** 9 final publication figures
- **Examples Archived:**
  - `training_curve_*.png` (exploratory)
  - `architecture_comparison_*.png` (intermediate)
  - `hpo_comparison_*.pdf` (drafts)

### From `reports/`
- **Period:** 2025-2026
- **Content:** Benchmark reports from earlier runs
- **Reason:** Superseded by final results
- **Location:** `archive/old_reports/`

### Old Documentation
- **MANU_MASSIVE_DOCUMENTATION_*.docx** → `archive/old_documentation/`
- **DOCUMENTATION_COMPLETE.md** → `archive/old_documentation/`
- **PROJECT_STRUCTURE.md** (old version) → `archive/old_documentation/`

### Deprecated Code
- **adme_gnn/** directory → `archive/old_code/`
- **Old model implementations** → `archive/old_code/`
- **Development scripts** → `archive/old_scripts/`

---

## 🚀 Git Integration

### After Reorganization

```bash
# See what changed
git status

# Stage new structure
git add results/ docs/

# Commit reorganization
git commit -m "refactor: reorganize project structure for clarity

- Move 37 HPO results to results/hpo/
- Copy 9 final publication figures to results/figures/
- Archive 100+ old files to archive/
- Create clean documentation in docs/
- Improve project navigation and clarity"

# Verify changes
git log --oneline -5
```

### .gitignore Recommendations

```gitignore
# Archive (large, unnecessary for git)
archive/

# Temporary files
*.tmp
*.log
*.pid

# Python
__pycache__/
*.pyc
.venv/

# Data (if using LFS)
*.npy
*.pkl
```

---

## 📋 Navigation Guide

### If you want to...

#### **Read the paper**
1. Open `paper_1/main.tex` in your LaTeX editor
2. View figures in `paper_1/images/`
3. Check results summary: `results/summary/FINAL_RESULTS_SUMMARY.md`

#### **Understand the methodology**
1. Read: `docs/README.md` (overview)
2. Read: `docs/METHODOLOGY.md` (experimental setup)
3. Read: `docs/DATASETS.md` (dataset details)

#### **Access experimental results**
1. HPO results: `results/hpo/[DATASET]/hpo_[DATASET]_[ALGORITHM].json`
2. Final figures: `results/figures/`
3. Summary statistics: `results/summary/FINAL_RESULTS_SUMMARY.md`

#### **Verify result authenticity**
1. Read: `docs/FORENSIC_ANALYSIS.md`
2. Check multi-seed files: `results/hpo/[DATASET]/hpo_*.json`
3. Review: Training histories in JSON files

#### **Run new experiments**
1. Install: `pip install -r requirements.txt`
2. Run: `python scripts/run_hpo_benchmark.py --dataset Caco2_Wang`
3. Results saved to: `results/hpo/Caco2_Wang/`

---

## 🔄 Structure Benefits

✅ **Clear separation** of final results from intermediate work
✅ **Easy navigation** - know exactly where to find things
✅ **Professional appearance** - clean, organized structure
✅ **Publication-ready** - structured for academic sharing
✅ **Archival** - old files preserved but out of the way
✅ **Reproducibility** - documentation explains everything
✅ **Scalability** - easy to add new experiments

---

**Last Updated:** 2026-03-09
**Reorganization Date:** 2026-03-09
**Status:** ✅ Complete & Ready for Publication
