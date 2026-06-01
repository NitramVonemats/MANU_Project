# Paper Results Navigation

This file maps the claims, tables, figures, and generated artifacts used by
`paper_final/main.tex`. It is meant as a quick guide for reviewers or project
members who want to verify where each paper result comes from.

## Quick Checks

Run the paper-number audit:

```powershell
py scripts\audit_paper_final_numbers.py
```

Expected result:

```text
TOTAL FAILURES: 0
```

Check whether there is any structured CSV/JSON evidence for 50 distinct random
seeds:

```powershell
py scripts\scan_seed_evidence.py
```

Expected result for the current repository:

```text
Files with >=50 distinct seeds: NONE
```

The paper uses **50 HPO trials** per algorithm/dataset and **5 multi-seed
validation seeds**. These are different checks.

Build the paper:

```powershell
cd paper_final
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Compiled output:

- `paper_final/main.pdf`

## Evidence Levels

- `Raw results`: JSON/CSV logs from runs or archived experiments exist.
- `Generated summary`: CSV/TEX/MD summary exists, but the full raw run logs may not be present.
- `Generated figure`: image/PDF artifact exists in `paper_final/images/` or `figures/`.
- `Derived evidence file`: JSON/CSV file exists in `results/`, reconstructed from detailed archived/documented values already present in the repo. This is not a new training run.

## Main Paper Files

| Purpose | Path |
|---|---|
| Paper source | `paper_final/main.tex` |
| Compiled paper | `paper_final/main.pdf` |
| References | `paper_final/refs.bib` |
| Paper images used by LaTeX | `paper_final/images/` |
| Reproducible audit script | `scripts/audit_paper_final_numbers.py` |
| Seed-evidence scanner | `scripts/scan_seed_evidence.py` |

## Tables

| Paper item | Location in paper | Evidence level | Source files | Notes |
|---|---:|---|---|---|
| Table I: related work summary | `paper_final/main.tex`, `tab:related_work_summary` | Literature/text | `paper_final/refs.bib` | Narrative/literature table, not project-generated numerical results. |
| Table II: HPO search space | `tab:search_space` | Raw config | `optimization/space.py` | Search choices match hidden dims, layers, LR, weight decay, and MLP heads. |
| Table III: dataset statistics | `tab:dataset_stats` | Raw data + generated summary | `datasets/adme/*.csv`, `datasets/toxicity/*.csv`, `figures/paper/dataset_statistics.csv` | TDC row counts and `Used` counts are both present. |
| Table IV: model-family comparison | `tab:model_comparison` | Raw/archived results | `archive/old_experiments/history/old_results/foundation_comparison_UPDATED_20260129_200243.csv`, `archive/old_experiments/history/old_results/foundation_comparison_COMPLETE_20260127_234555.csv`, `runs/*/*.json` | GNN, ChemBERTa, Morgan-FP, MolE-FP, and MolCLR values are traceable. |
| Table V: regression architecture comparison | `tab:arch_regression` | Raw/archived summary | `archive/old_experiments/MODEL_STATISTICS.csv`, `archive/old_experiments/history/old_docs/docs/MANU_Complete_Documentation.md` | Values match architecture summary data. |
| Table VI: classification architecture comparison | `tab:arch_classification` | Derived evidence file + generated summary | `figures/paper-sources-2/gnn_classification_architecture_results.csv`, `figures/paper-sources-2/gnn_classification_architecture_results.json`, `scripts/generate_gnn_architecture_comparison.py`, `archive/summaries/РЕЗИМЕ_ПОДГОТВЕНО.md`, `archive/old_experiments/history/old_docs/docs/MANU_Complete_Documentation.md` | Structured CSV/JSON evidence now exists, derived from archived documented architecture results. Full raw classification architecture training logs are still not present as separate JSON runs. |
| Table VII: architecture ranking | `tab:arch_ranking` | Generated summary | `scripts/generate_gnn_architecture_comparison.py` | Ranking values match the script used to generate the architecture summary figure. Note: `docs/DOCUMENTATION.md` contains an older conflicting appendix ranking. |
| Table VIII: HPO algorithm comparison | `tab:hpo_results` | Raw results | `runs/*/hpo_*.json`, `archive/old_experiments/history/old_results/tpe_*_results.json` | NiaPy algorithms come from `runs/`; TPE comes from archived Optuna JSONs. |
| Table IX: best NiaPy result per dataset | `tab:best_results` | Raw results | `runs/*/hpo_*.json` | NiaPy-only table; intentionally excludes TPE because TPE has a slightly larger search space. |
| Table X: HPO vs Random confidence intervals | `tab:ci_comparison` | Generated summary | `figures/paper-sources-2/hpo_ci_summary.csv`, `figures/paper-sources-2/hpo_paired_improvements.csv`, `scripts/hpo_confidence_intervals.py` | CI values are generated and audit-checked. |
| Table XI: multi-seed validation | `tab:multiseed` | Derived evidence file + generated figure | `figures/paper-sources-2/multi_seed_results_fixed.json`, `figures/paper-sources-2/multi_seed_raw_results_fixed.csv`, `figures/paper-sources-2/multi_seed_summary_fixed.csv`, `docs/DOCUMENTATION.md`, `figures/paper/multiseed_table.tex`, `paper_final/images/multi_seed_validation.png` | Structured JSON/CSV evidence now exists, reconstructed from the detailed 5-seed values in `docs/DOCUMENTATION.md` Appendix H. |

## Figures

| Paper figure | Image used by paper | Evidence level | Source or generation path | Notes |
|---|---|---|---|---|
| Benchmark framework | `paper_final/images/Benchmark-Architecture-GNN.png` | Generated figure | `paper_final/images/Benchmark-Architecture-GNN.png` | Diagram artifact exists. |
| Data origin / class distribution | `paper_final/images/sankey-diagram.png` | Generated figure + raw data | `datasets/*/*.csv`, `figures/paper/dataset_statistics.csv` | Counts and class balance are audit-checked. |
| Foundation ranking | `paper_final/images/foundation_ranking.png` | Generated figure + archived results | `archive/old_experiments/history/old_results/foundation_comparison_UPDATED_20260129_200243.csv` | Figure exists and values trace to archived result CSVs. |
| GNN vs foundation comparison | `paper_final/images/gnn_vs_foundation_comparison.png` | Generated figure + archived/raw results | `scripts/regenerate_paper_figures.py`, `archive/old_experiments/history/old_results/foundation_comparison_UPDATED_20260129_200243.csv`, `runs/*/*.json` | Figure exists. |
| Architecture comparison | `paper_final/images/gnn_architecture_comparison_all_datasets.png` | Generated figure | `scripts/generate_gnn_architecture_comparison.py`, `archive/old_experiments/MODEL_STATISTICS.csv` | Figure exists. Classification panel values are generated from summary values. |
| Architecture selection summary | `paper_final/images/gnn_architecture_selection_summary.png` | Generated figure | `scripts/generate_gnn_architecture_comparison.py` | Figure exists. |
| HPO convergence curves | `paper_final/images/hpo_convergence_curves.png` | Generated figure + raw runs | `runs/*/hpo_*.json`, `scripts/regenerate_paper_figures.py` | Figure exists. |
| Algorithm performance | `paper_final/images/01_algorithm_performance.png` | Generated figure + raw runs | `runs/*/hpo_*.json`, `scripts/regenerate_paper_figures.py` | Figure exists. |
| Confusion matrices | `paper_final/images/confusion_matrices.png` | Generated figure + raw runs | `runs/tox21/hpo_tox21_sa.json`, `runs/herg/hpo_herg_abc.json`, `scripts/generate_confusion_matrices.py` | Figure exists. |
| Multi-seed validation | `paper_final/images/multi_seed_validation.png` | Generated figure + documented values | `docs/DOCUMENTATION.md`, `scripts/regenerate_paper_figures.py` | Figure exists; fixed raw 5-seed JSON is not present. |

## Key Claims And Where To Verify Them

| Claim in paper | Verification source | Status |
|---|---|---|
| Tox21 has 7,258 TDC compounds and 6,533 used after graph conversion | `datasets/toxicity/Tox21.csv`, `figures/paper/dataset_statistics.csv` | Verified |
| Tox21 positive rate is about 4.2% | `datasets/toxicity/Tox21.csv` | Verified |
| hERG has 655 TDC compounds and 590 used after graph conversion | `datasets/toxicity/hERG.csv`, `figures/paper/dataset_statistics.csv` | Verified |
| hERG blocker split is about 69/31 | `datasets/toxicity/hERG.csv` | Verified |
| GNN hERG AUC = 0.825 | `runs/herg/hpo_herg_abc.json` | Verified |
| GNN Caco2 R2 = 0.481 | `runs/Caco2_Wang/hpo_Caco2_Wang_random.json` | Verified |
| GNN Tox21 AUC = 0.742 | `runs/tox21/hpo_tox21_sa.json` | Verified |
| No HPO metaheuristic improves over Random Search on average | `figures/paper-sources-2/hpo_ci_summary.csv` | Verified |
| Multi-seed means and CIs | `figures/paper-sources-2/multi_seed_results_fixed.json`, `figures/paper-sources-2/multi_seed_summary_fixed.csv`, `docs/DOCUMENTATION.md`, `figures/paper/multiseed_table.tex` | Verified as structured evidence derived from documented per-seed values |
| Architecture classification values | `figures/paper-sources-2/gnn_classification_architecture_results.csv`, `figures/paper-sources-2/gnn_classification_architecture_results.json`, `scripts/generate_gnn_architecture_comparison.py`, archived docs | Verified as structured evidence derived from archived architecture summaries |

## Raw Result Locations

### Current HPO runs

```text
runs/Caco2_Wang/
runs/Half_Life_Obach/
runs/Clearance_Hepatocyte_AZ/
runs/Clearance_Microsome_AZ/
runs/tox21/
runs/herg/
```

Each folder contains one JSON per NiaPy algorithm:

```text
hpo_<dataset>_pso.json
hpo_<dataset>_abc.json
hpo_<dataset>_ga.json
hpo_<dataset>_sa.json
hpo_<dataset>_hc.json
hpo_<dataset>_random.json
```

### TPE results

```text
archive/old_experiments/history/old_results/tpe_Caco2_Wang_results.json
archive/old_experiments/history/old_results/tpe_Half_Life_Obach_results.json
archive/old_experiments/history/old_results/tpe_Clearance_Hepatocyte_AZ_results.json
archive/old_experiments/history/old_results/tpe_Clearance_Microsome_AZ_results.json
archive/old_experiments/history/old_results/tpe_tox21_results.json
archive/old_experiments/history/old_results/tpe_herg_results.json
```

### Foundation/pretrained comparison

```text
archive/old_experiments/history/old_results/foundation_comparison_UPDATED_20260129_200243.csv
archive/old_experiments/history/old_results/foundation_comparison_COMPLETE_20260127_234555.csv
archive/old_experiments/history/old_results/molclr_pretrained_results_20260129_200103.csv
```

### Architecture comparison

```text
figures/paper-sources-2/gnn_classification_architecture_results.csv
figures/paper-sources-2/gnn_classification_architecture_results.json
archive/old_experiments/MODEL_STATISTICS.csv
scripts/generate_gnn_architecture_comparison.py
archive/summaries/РЕЗИМЕ_ПОДГОТВЕНО.md
archive/old_experiments/history/old_docs/docs/MANU_Complete_Documentation.md
```

### Multi-seed validation

```text
figures/paper-sources-2/multi_seed_results_fixed.json
figures/paper-sources-2/multi_seed_raw_results_fixed.csv
figures/paper-sources-2/multi_seed_summary_fixed.csv
docs/DOCUMENTATION.md
figures/paper/multiseed_table.tex
paper_final/images/multi_seed_validation.png
```

Important: `archive/old_experiments/history/old_results/multiseed/` contains a 3-seed CSV/JSON set with seeds `42,43,44`; it is not the source of the 5-seed Table XI. The 5-seed JSON/CSV files in `figures/paper-sources-2/` were reconstructed from the documented per-seed values in `docs/DOCUMENTATION.md` Appendix H.

No CSV/JSON file in the repository contains 50 distinct random seeds. Files that
mention `50` are HPO trial-budget files (`50 trials`), not `50 seeds`.

## Generated Artifacts That Exist

These paper-facing artifacts are present:

```text
paper_final/main.pdf
paper_final/images/Benchmark-Architecture-GNN.png
paper_final/images/sankey-diagram.png
paper_final/images/foundation_ranking.png
paper_final/images/gnn_vs_foundation_comparison.png
paper_final/images/gnn_architecture_comparison_all_datasets.png
paper_final/images/gnn_architecture_selection_summary.png
paper_final/images/hpo_convergence_curves.png
paper_final/images/01_algorithm_performance.png
paper_final/images/confusion_matrices.png
paper_final/images/multi_seed_validation.png
figures/paper-sources-2/hpo_ci_summary.csv
figures/paper-sources-2/hpo_paired_improvements.csv
figures/paper/dataset_statistics.csv
figures/paper-sources-2/gnn_classification_architecture_results.csv
figures/paper-sources-2/gnn_classification_architecture_results.json
figures/paper-sources-2/multi_seed_results_fixed.json
figures/paper-sources-2/multi_seed_raw_results_fixed.csv
figures/paper-sources-2/multi_seed_summary_fixed.csv
```

## Known Caveats

1. Table XI now has JSON/CSV evidence under `figures/paper-sources-2/`, but those files are reconstructed from documented per-seed values, not recovered original training logs.
2. Table VI now has JSON/CSV evidence under `figures/paper-sources-2/`, but those files are reconstructed from archived summary values, not recovered original per-run training logs.
3. Table VII ranking values match `scripts/generate_gnn_architecture_comparison.py`; `docs/DOCUMENTATION.md` has an older conflicting appendix ranking and should not be used as the source for the paper's current Table VII.
4. Some files in `figures/paper/` are stale generated tables from an older version of the paper. Prefer `paper_final/main.tex`, `scripts/audit_paper_final_numbers.py`, and the sources listed above for the current version.
5. A full project scan found no structured 50-seed result file. The validated
   paper claim is five seeds `[42, 123, 456, 789, 1011]`, plus 50 HPO trials per
   algorithm/dataset.
