# Reproduction guide — *Task-Dependent Performance of GNNs and Pretrained Models in ADMET Prediction*

Every number in `main.tex` / `supplementary_materials.tex` traces to one committed
artifact and one generator script. This directory collects the load-bearing
artifacts (some of which otherwise live under the git-ignored `results/` tree) and
documents how each table and figure is produced.

## Table → artifact → generator map

| Paper element | Committed artifact | Generator | Notes |
|---|---|---|---|
| **Table III** dataset stats (TDC / Used counts) | `datasets/adme/*.csv`, `datasets/toxicity/*.csv` (raw TDC counts) + `artifacts/model_comparison/table_IV_model_comparison.csv` (`train_size`+`test_size` = Used count) | `optimized_gnn.py` `prepare_dataset()` (scaffold split + invalid-SMILES filter) | Used counts: Caco2 637+182=819, Half-Life 466+135=601, Clear-Hep 849+243=1092, Clear-Mic 771+221=992, Tox21 5080+1453=6533, hERG 458+132=590 |
| **Table IV** model-family comparison (30 cells) | `artifacts/model_comparison/table_IV_model_comparison.csv` | `scripts/run_complete_foundation_benchmark.py` (+ `optimization/foundation_*.py`) | All 30 cells match the paper to 3 dp. Not turnkey to rerun: needs HuggingFace ChemBERTa weights + local MolCLR checkpoint + network; the committed CSV is the authoritative record. |
| **Table V/VI/VII** HPO per-optimizer, best-per-dataset, F1 | `runs/<dataset>/hpo_<dataset>_<algo>.json` → `final_training.test_metrics` | `scripts/run_gpu_hpo_extended.py` | `runs/` is git-tracked. 6 optimizers × 6 datasets. |
| **Table (HPO vs Random, bootstrap CI)** `tab:ci_comparison` | derived from `runs/*.json` | `hpo_ci_analysis.py` | Reproduces every CI, d_z, W/T/L exactly (classification metric = F1). |
| **Table (imbalance metrics)** `tab:imbalance_metrics` + ECE + toxicity figures | `artifacts/toxicity_extended/toxicity_extended_metrics.json` (+ `results/toxicity_extended/predictions/*.npz`) | `toxicity_extended_metrics.py` → `make_toxicity_figures.py` | ECE 0.020/0.067 recomputed from pooled predictions. |
| **Table (multi-seed)** `tab:multiseed` | `artifacts/multi_seed/multi_seed_results_fixed.json` | `scripts/run_multi_seed_validation.py` | Seeds [42,123,456,789,1011]; best-HPO config per dataset (hardcoded from `runs/*.json` best trials). Regression metric = original-scale RMSE (`rmse_orig_*`); 95% CI = mean ± 1.96·std/√5. |
| **Supp. Tables S1/S2 + arch figures** | `artifacts/architecture_selection/architecture_selection_search.json`, `table_S1_classification.tex`, `table_S2_regression_ranking.tex` | `scripts/run_architecture_selection.py --mode search` → `scripts/make_architecture_figures.py` | 8 architectures × 6 endpoints × best-of-5 configs (240 runs). |

## Regenerating from scratch

```bash
# 1. HPO sweep (produces runs/*.json → Tables V–VII)
python scripts/run_gpu_hpo_extended.py

# 2. Multi-seed validation (→ results/multi_seed/ → multi-seed table)
python scripts/run_multi_seed_validation.py

# 3. Architecture-selection screen (→ results/architecture_selection/ → S1/S2)
python scripts/run_architecture_selection.py --mode search --epochs 50 --patience 12
python scripts/make_architecture_figures.py

# 4. Imbalance-aware toxicity metrics + figures (→ results/toxicity_extended/)
python paper_final/reproduction/toxicity_extended_metrics.py
python paper_final/reproduction/make_toxicity_figures.py

# 5. HPO vs Random bootstrap-CI table
python paper_final/reproduction/hpo_ci_analysis.py

# 6. Foundation-model baselines for Table IV (NOT turnkey — needs weights/network)
#    The committed artifacts/model_comparison/table_IV_model_comparison.csv is authoritative.
```

Core training model shared by all runners: `optimized_gnn.py` (repo root).
Environment: Python ≥3.8, PyTorch ≥2.0, PyTorch Geometric, RDKit, PyTDC, NiaPy, Optuna, Transformers.

## Known reproducibility caveats

- **Table IV is not bit-for-bit turnkey**: frozen ChemBERTa needs HuggingFace weights
  (`seyonec/ChemBERTa-zinc-base-v1`), MolCLR needs the local GIN checkpoint under
  `external/MolCLR/ckpt/`, and untrained projection heads (Morgan-FP/MolE-FP) are only
  reproducible within the same torch/sklearn build. The committed CSV is the record of record.
- **MolE-FP is a placeholder**: Morgan fingerprints + a learned projection inspired by MolE,
  **not** the pretrained MolE encoder (disclosed in the paper, Section III-C).
- **Descriptor asymmetry**: only the GNN receives the auxiliary structure-derived
  physicochemical descriptors; the pretrained/fingerprint baselines do not (disclosed in
  Methods and Limitations).
