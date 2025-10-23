#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze optimized GNN benchmark CSVs and print summaries.

Looks for files named: optimized_gnn_results_*.csv
Usage:
  python scripts/analyze_gnn_results.py [search_root]

If search_root is omitted, current directory is used.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# Where to search (relative to 'root')
SEARCH_PATTERNS = [
    "optimized_gnn_results_*.csv",     # repo root
    "**/optimized_gnn_results_*.csv",  # anywhere under root
    "results/**/*.csv",                # common folder
    "GNN_test/**/*.csv",               # legacy folder (kept for completeness)
]

# Datasets you care about
DATASETS = [
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "Caco2_Wang",
]

def find_csvs(root: str) -> list[str]:
    root = Path(root).resolve()
    seen = set()
    files: list[str] = []
    for pat in SEARCH_PATTERNS:
        for f in glob.glob(str(root / pat), recursive=True):
            p = str(Path(f).resolve())
            if p.lower().endswith(".csv") and p not in seen:
                seen.add(p)
                files.append(p)
    return sorted(files)

def load_results_file(path: str) -> pd.DataFrame:
    """
    Expected columns from optimized_gnn.py:
      Dataset, Val_RMSE, Test_RMSE, Test_MAE, Test_R2, Time_s
    We keep any subset of those (but require Dataset & Test_RMSE).
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️  Skip unreadable CSV: {path} ({e})")
        return pd.DataFrame()

    # Harmonize possible lowercase/alternate names
    colmap = {c.lower(): c for c in df.columns}
    need_ds = "dataset" in colmap
    need_rmse = "test_rmse" in colmap
    if not (need_ds and need_rmse):
        print(f"⚠️  Skip incompatible CSV (needs 'Dataset' and 'Test_RMSE'): {path}")
        return pd.DataFrame()

    # Preferred columns to keep if present
    keep_order = ["Dataset", "Val_RMSE", "Test_RMSE", "Test_MAE", "Test_R2", "Time_s"]
    present = [k for k in keep_order if k in df.columns]
    out = df[present].copy()

    # Make numerics numeric
    for col in ["Val_RMSE", "Test_RMSE", "Test_MAE", "Test_R2", "Time_s"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["source_file"] = os.path.basename(path)
    out["source_path"] = path
    return out

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print("DEBUG: CWD  =", os.getcwd())
    print("DEBUG: root =", Path(root).resolve())
    csv_files = find_csvs(root)

    print(f"Пронајдени {len(csv_files)} CSV фајлови")
    for f in csv_files:
        print(" -", f)

    # Early exit if nothing found
    if not csv_files:
        print("\n❌ Нема резултатски CSV фајлови.")
        print("   Совети:")
        print("   • Прво пушти го бенчмаркот: python src\\optimized_gnn.py")
        print("   • Провери каде се сними фајлот (скриптата печати „💾 Results saved: ...“ на крај).")
        print("   • Ако CSV е под src/, стартувај:  python scripts\\analyze_gnn_results.py src")
        sys.exit(1)

    # Load all
    all_df = []
    for path in csv_files:
        df = load_results_file(path)
        if not df.empty:
            all_df.append(df)

    if not all_df:
        print("\n❌ Не вчитав ниту еден валиден CSV (празни/несоодветни колони).")
        sys.exit(1)

    combined_df = pd.concat(all_df, ignore_index=True)
    print(f"\nВкупно вчитани записи: {len(combined_df)} од {len(all_df)} валидни CSV фајлови.\n")

    # ================= EXECUTIVE SUMMARY =================
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY – НАЈДОБРИ РЕЗУЛТАТИ ПО DATASET (минимален Test_RMSE)")
    print("=" * 100)

    summary_rows = []
    for ds in DATASETS:
        df_ds = combined_df[combined_df["Dataset"] == ds]
        if df_ds.empty:
            print(f"\n{ds}: (нема записи)")
            continue

        best = df_ds.loc[df_ds["Test_RMSE"].idxmin()]
        print(f"\n{ds}:")
        print(f"  Source:     {best.get('source_file', 'unknown')}")
        print(f"  Test_RMSE:  {best['Test_RMSE']:.4f}")
        if "Test_R2" in df_ds.columns:
            print(f"  Test_R2:    {best['Test_R2']:.4f}")
        if "Val_RMSE" in df_ds.columns:
            print(f"  Val_RMSE:   {best['Val_RMSE']:.4f}")

        summary_rows.append({
            "Dataset": ds,
            "Best Test_RMSE": best["Test_RMSE"],
            "Best Test_R2": best.get("Test_R2", np.nan),
            "From file": best.get("source_file", "unknown"),
        })

    if summary_rows:
        print("\n" + "=" * 100)
        print("РЕЗИМЕ ТАБЕЛА")
        print("=" * 100)
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

    # ================= ALL RESULTS (ranked) =================
    print("\n" + "=" * 100)
    print("СИТЕ РЕЗУЛТАТИ – по dataset и фајл (сортирано по Test_RMSE)")
    print("=" * 100)
    ranked = combined_df.sort_values(["Dataset", "Test_RMSE", "source_file"])
    display_cols = [c for c in ["Dataset", "Val_RMSE", "Test_RMSE", "Test_MAE", "Test_R2", "Time_s", "source_file"] if c in ranked.columns]
    print(ranked[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
