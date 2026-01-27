# Foundation Models Benchmark - Running

**Started:** 2025-12-15
**Status:** 🟢 RUNNING

---

## Што се извршува:

### Datasets (7):
1. **Caco2_Wang** (910 molecules) - Regression
2. **Half_Life_Obach** (667 molecules) - Regression
3. **Clearance_Hepatocyte_AZ** (1,213 molecules) - Regression
4. **Clearance_Microsome_AZ** (1,102 molecules) - Regression
5. **tox21** (7,258 molecules) - Classification
6. **herg** (655 molecules) - Classification
7. **clintox** (1,478 molecules) - Classification

**Total:** 13,283 molecules

### Foundation Models (2):
1. **Morgan Fingerprint** (baseline - ECFP4, 2048 bits)
2. **ChemBERTa** (Transformer for SMILES)

**Note:** BioMed, MolCLR, MolE се skip-нати (бараат дополнителен setup)

---

## Expected Output:

### For Regression (ADME):
- Test RMSE
- Test R²
- Test MAE
- Val RMSE, Val R²

### For Classification (Tox):
- Test AUC-ROC
- Test Accuracy
- Test F1
- Val AUC, Val Accuracy

---

## Where to Find Results:

### Results File:
```
results/foundation_benchmark/benchmark_results_YYYYMMDD_HHMMSS.csv
```

### Log File:
```
foundation_benchmark.log
```

### Monitor Progress:
```bash
tail -f foundation_benchmark.log
```

---

## Estimated Time:

- **Morgan Fingerprint:** Fast (few seconds per dataset)
- **ChemBERTa:** Slow (1-2 minutes per dataset for embedding extraction)

**Total estimated time:** 10-15 minutes for all 7 datasets

---

## What Happens After:

1. ✅ Results saved to CSV
2. ⏳ Create comparison with GNN results
3. ⏳ Generate visualization (GNN vs Foundation)
4. ⏳ Update benchmark report

---

## Progress:

Check `foundation_benchmark.log` for real-time progress:
```bash
tail -f foundation_benchmark.log
```

Or check results directory:
```bash
ls -la results/foundation_benchmark/
```

---

**Note:** Ова трае малку време. Оди пиј кафе ☕ и врати се за 10-15 минути!
