# MANU_Project — Целосна ревизија на податоци и документација

**Датум:** 2026-03-22  
**Repo:** https://github.com/NitramVonemats/MANU_Project.git

---

## 1. ИЗВОР НА ВИСТИНА: Актуелни резултати од JSON фајлови

### 1.1 GNN HPO (6 алгоритми × 6 датасети, 50 trials) — `runs/`

**ADME Regression (Test RMSE — пониско = подобро)**

| Dataset | Random | PSO | ABC | GA | SA | HC | **Best** |
|---------|--------|-----|-----|----|----|----|----|
| Caco2_Wang | **0.00271** | 0.00310 | 0.00290 | 0.00310 | 0.00288 | 0.00300 | Random |
| Half_Life_Obach | 22.315 | **21.658** | **21.658** | **21.658** | 23.695 | 24.524 | PSO=ABC=GA |
| Clearance_Hepatocyte_AZ | **68.216** | 70.206 | 72.042 | 71.345 | 72.042 | 72.042 | Random |
| Clearance_Microsome_AZ | **38.751** | 42.759 | 42.288 | 42.288 | 40.940 | 41.635 | Random |

**Toxicity Classification (Test AUC-ROC — повисоко = подобро)**

| Dataset | Random | PSO | ABC | GA | SA | HC | **Best** |
|---------|--------|-----|-----|----|----|----|----|
| Tox21 (NR-AR) | 0.713 | 0.692 | 0.735 | 0.735 | **0.742** | 0.652 | SA |
| hERG | 0.747 | 0.747 | **0.825** | 0.747 | 0.802 | 0.821 | ABC |

> ✅ **Овие вредности се ВЕРИФИЦИРАНИ** директно од JSON фајловите во `runs/`.

### 1.2 TPE Benchmark — `archive/old_experiments/history/old_results/tpe_*_results.json`

| Dataset | test_rmse_orig | test_rmse_log | test_auc |
|---------|---------------|---------------|----------|
| Caco2_Wang | 0.00290 | 0.519 | — |
| Half_Life_Obach | 21.478 | 1.152 | — |
| Clearance_Hepatocyte_AZ | **80.316** | 1.339 | — |
| Clearance_Microsome_AZ | 40.887 | 1.198 | — |
| Tox21 | — | — | 0.722 |
| hERG | — | — | 0.756 |

### 1.3 ChemBERTa-FT — `archive/old_results/chemberta_ft_*_results.json`

| Dataset | test_rmse_orig | test_rmse_log | test_auc |
|---------|---------------|---------------|----------|
| Caco2_Wang | 0.00323 | 0.500 | — |
| Half_Life_Obach | **8.311** | 1.066 | — |
| Clearance_Hepatocyte_AZ | 52.597 | 1.417 | — |
| Clearance_Microsome_AZ | 42.873 | 1.289 | — |
| Tox21 | — | — | **0.482** |
| hERG | — | — | 0.777 |

### 1.4 Foundation Models (Frozen) — `archive/.../foundation_comparison_UPDATED_*.csv`

| Model | Caco2 RMSE | Half_Life RMSE | Clear_Hep RMSE | Clear_Micro RMSE | Tox21 AUC | hERG AUC |
|-------|-----------|---------------|---------------|-----------------|-----------|----------|
| Morgan-FP | 0.614 | 22.12 | 48.36 | 40.36 | 0.722 | 0.611 |
| ChemBERTa (frozen) | 0.496 | 27.39 | 47.31 | 42.56 | 0.728 | 0.770 |
| MolE-FP | 0.670 | 25.01 | 47.22 | 41.79 | 0.675 | 0.672 |
| MolCLR | 0.749 | 21.71 | 48.92 | 42.19 | 0.452 | 0.401 |

> ⚠️ **ВАЖНО:** Foundation model RMSE за Caco2 е во **log-space** (~0.5-0.75), додека GNN RMSE е во **original-space** (~0.0027). Овие НЕ МОЖЕ директно да се споредуваат!

### 1.5 Multi-seed — `archive/old_results/multi_seed_results_BUGGY.json`

| Dataset | Task | Mean rmse_orig ± Std |
|---------|------|---------------------|
| Caco2_Wang | Regression | 0.00322 ± 0.00040 |
| Half_Life_Obach | Regression | 37.81 ± 43.45 |
| Clearance_Hepatocyte_AZ | Regression | — |
| Clearance_Microsome_AZ | Regression | — |
| Tox21 | Classification | — |
| hERG | Classification | — |

> ⚠️ Фајлот е самоименуван **"BUGGY"** — овие податоци не се веродостојни без ревалидација.

---

## 2. ГРЕШКИ ВО README.md

### 2.1 🔴 КРИТИЧНО: TPE колона — сите вредности се погрешни

| Dataset | README TPE | JSON orig | JSON log | Грешка |
|---------|-----------|-----------|----------|--------|
| Caco2_Wang | 0.526 | 0.00290 | 0.519 | README го користи log-RMSE наместо orig-RMSE |
| Half_Life | **98.47** | 21.478 | 1.152 | Не одговара на НИТУ ЕДНА метрика |
| Clear_Hep | **47.52** | 80.316 | 1.339 | Не одговара — README тврди TPE е best, но orig=80.32 е WORST |
| Clear_Micro | 39.04 | 40.887 | 1.198 | Блиску но не точно |
| Tox21 AUC | 0.742 | 0.722 | — | Погрешна вредност (+0.02) |
| hERG AUC | 0.745 | 0.756 | — | Погрешна вредност (-0.01) |

**Импликација:** README тврди дека TPE е best на Clearance_Hepatocyte (47.52 vs 68.22), но реалните JSON податоци покажуваат TPE = **80.316** (најлоши!). Ова го менува целиот наратив за TPE.

### 2.2 🔴 КРИТИЧНО: ChemBERTa-FT колона

| Dataset | README | JSON actual | Грешка |
|---------|--------|-------------|--------|
| Caco2 | 0.506 | 0.00323 orig / 0.500 log | Мешана скала |
| Half_Life | **21.99** | **8.311** orig / 1.066 log | Целосно погрешно |
| Clear_Hep | 49.39 | 52.597 | Погрешно (-3.2) |
| Clear_Micro | 43.25 | 42.873 | Блиску |
| **Tox21 AUC** | **0.735** | **0.482** | 🚨 КРИТИЧНО: Реално полошо од random (0.5)! |
| hERG AUC | 0.791 | 0.777 | Погрешно (+0.014) |

**Импликација:** README тврди ChemBERTa-FT Tox21 AUC = 0.735 (солидно), реалност = **0.482** (полошо од coin flip). Ова е наодот за "catastrophic overfitting" кој е коректно спомнат во paper abstract но погрешно во README табелата.

### 2.3 🔴 Foundation Model Comparison — мешање на скали

README ги споредува:
- **GNN-Best Caco2 = 0.0027** (original permeability units)
- **Morgan-FP Caco2 = 0.614** (log-space RMSE)

Ова е споредба на *две различни метрики*. Читателот помислува GNN е 200x подобро, но всушност се различни единици.

**Решение:** Или конвертирај сè во иста скала, или експлицитно означи ги единиците.

### 2.4 🟡 Winner Summary — неточности

README Winner Summary:

| README тврди | Реалност |
|-------------|----------|
| "Random wins 2/6" (Caco2, Clear_Micro) | Random wins **3/6** (+ Clear_Hep со RMSE=68.22, best од NiaPy) |
| "PSO wins 1/6 Half_Life (tie with ABC, GA)" | Точно — сите три имаат идентичен RMSE=21.658 |
| "TPE wins 1/6 Clear_Hep (47.52 vs 68.22)" | ❌ НЕТОЧНО — TPE orig=80.32 е ПОЛОШО |

### 2.5 🟡 Key Findings — неточни тврдења

| README Finding | Реалност |
|---------------|----------|
| "TPE excels on complex clearance tasks — Best on Clearance_Hepatocyte (47.52 vs 68.22 RMSE)" | ❌ TPE orig RMSE=80.32, **НАЈЛОШО** на Clear_Hep |
| "ChemBERTa fine-tuning improves toxicity prediction — AUC 0.79 on hERG, 0.73 on Tox21" | ❌ hERG=0.777 (блиску), **Tox21=0.482** (catastrophic fail) |
| "50 trials is sufficient — Diminishing returns beyond this budget" | ⚠️ Не е директно поддржано со податоци во repo |

### 2.6 Multi-seed табела во README

| README | JSON файлови |
|--------|-------------|
| Caco2: 0.631 ± 0.065 | rmse_log mean=0.564 ± 0.079 / rmse_orig=0.00322 ± 0.0004 |
| Half_Life: 42.82 ± 41.09 | rmse_orig=37.81 ± 43.45 |

README вредностите не одговараат на JSON (дури ни приближно за Caco2).

---

## 3. ГРЕШКИ ВО ДОКУМЕНТАЦИЈАТА

### 3.1 Референци кон непостоечки фајлови

| Документ | Реферира | Постои? |
|----------|---------|---------|
| README.md | `DOCUMENTATION.md` | ❌ |
| README.md | `docs/STATUS/` | ❌ |
| README.md | `figures/hpo/01_algorithm_performance.png` | ❌ |
| README.md | `figures/hpo/03_winner_analysis.png` | ❌ |
| README.md | `figures/hpo/05_classification_performance.png` | ❌ |
| README.md | `figures/paper/foundation_comparison_with_finetune.png` | ❌ (.pdf постои) |
| README.md | `LICENSE` файл | ❌ |
| docs/README.md | `docs/FORENSIC_ANALYSIS.md` | ❌ |
| docs/README.md | `results/` директориум | ❌ |
| docs/README.md | `results/hpo/` | ❌ (податоците се во `runs/`) |
| docs/README.md | `results/figures/` | ❌ |
| docs/README.md | `results/summary/FINAL_RESULTS_SUMMARY.md` | ❌ |
| docs/README.md | `code/` директориум | ❌ (кодот е во `src/`) |
| docs/METHODOLOGY.md | `scripts/run_hpo_benchmark.py` | ❌ |
| docs/PROJECT_STRUCTURE.md | Целата структура `results/`, `code/` | ❌ Никогаш реорганизирано |

### 3.2 Неконзистентност меѓу документите

| Тема | README.md | docs/README.md | Paper |
|------|-----------|---------------|-------|
| Random wins | 2/6 | 3/4 regression | 2 datasets (NiaPy) |
| TPE Clear_Hep | 47.52 (best) | не спомнато | 52.16 |
| Total compute | ~40 hours | ~45 hours | не спомнато |
| Datasets | "6 (4 ADME + 2 Toxicity)" | Исто | Исто ✅ |

### 3.3 Paper LaTeX vs JSON податоци

Paper `hpo_results_table.tex` покажува **СТАРИ** бројки:

| Dataset/Algo | Paper .tex | Actual JSON |
|-------------|-----------|-------------|
| Caco2 ABC | 0.0026 | 0.00290 |
| Caco2 PSO | 0.0026 | 0.00310 |
| Clear_Hep SA | 50.29 | 72.04 |
| Clear_Micro SA | 40.86 | 40.94 |

Paper main.tex HPO табела (Table 4) ги има **точните** бројки од JSON! Значи `paper/tables/hpo_results_table.tex` е outdated а `paper_1/main.tex` е ажурирана.

---

## 4. СТРУКТУРНИ ПРОБЛЕМИ

### 4.1 Project Structure — реалност vs документација

**Реална структура:**
```
MANU_Project/
├── src/                    ← Source code (NOT code/)
├── runs/                   ← HPO results (NOT results/hpo/)
├── optimization/           ← HPO algorithms
├── figures/paper/          ← Paper figures (PDFs, not PNGs)
├── paper_1/                ← LaTeX paper ✅
├── paper/                  ← Documentation PDF + tables
├── docs/                   ← Documentation (partially incomplete)
├── archive/                ← Old files ✅
├── scripts/                ← Scripts ✅
├── datasets/               ← Raw data ✅
└── README.md
```

**Документирана структура (PROJECT_STRUCTURE.md):**
```
results/hpo/      ← НЕ ПОСТОИ
results/figures/  ← НЕ ПОСТОИ
results/summary/  ← НЕ ПОСТОИ
code/             ← НЕ ПОСТОИ
```

### 4.2 Липсуваат фајлови

- `LICENSE` — README вели MIT но нема фајл
- `.gitignore` — не проверен
- `DOCUMENTATION.md` — реферирано но не постои

---

## 5. PAPER-SPECIFIC ПРОБЛЕМИ (paper_1/main.tex)

### 5.1 TPE табела во paper

Paper Table 2 (TPE standalone):
| Dataset | Paper | JSON orig |
|---------|-------|-----------|
| Clear_Hep | 52.16 | **80.316** |
| Clear_Micro | 44.34 | 40.887 |
| Tox21 AUC | 0.705 | 0.722 |
| hERG AUC | 0.772 | 0.756 |

**Ниту една TPE вредност во paper не одговара на JSON!** Ова значи или: (а) постои друг TPE run чии резултати не се во repo, или (б) бројките се внесени рачно и погрешно.

### 5.2 Автори

Paper: "Martin, Mila, Adrian, Viktorija, Ilinka" — без презимиња. За публикација треба полни имиња и афилијации.

### 5.3 ChemBERTa overfitting — ТОЧНО

Paper коректно тврди: "ChemBERTa-FT Tox21: validation AUC = 0.82, test AUC = 0.46". JSON потврдува test_auc = 0.482. Ова е силен и валиден наод.




*Генерирано автоматски на 2026-03-22*
