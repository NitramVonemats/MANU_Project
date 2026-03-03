"""
Quick comparison script - analyze existing CSV results
Use this to quickly compare results from primer.py vs primer_v2.py
"""
import pandas as pd
import numpy as np
from glob import glob
import os

def find_latest_results(prefix):
    """Find latest CSV files for given prefix"""
    files = glob(f"{prefix}_*.csv")
    if not files:
        return None
    # Get the most recent file
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def compare_results():
    print("\n" + "="*80)
    print("  QUICK COMPARISON: primer.py vs primer_v2.py")
    print("="*80 + "\n")

    # Find baseline results
    baseline_files = {
        "Half_Life_Obach": find_latest_results("fixed_molecular_gnn_Half_Life_Obach"),
        "Clearance_Hepatocyte_AZ": find_latest_results("fixed_molecular_gnn_Clearance_Hepatocyte_AZ"),
        "Clearance_Microsome_AZ": find_latest_results("fixed_molecular_gnn_Clearance_Microsome_AZ"),
    }

    # Find v2 results
    v2_files = {
        "Half_Life_Obach": find_latest_results("primer_v2_Half_Life_Obach"),
        "Clearance_Hepatocyte_AZ": find_latest_results("primer_v2_Clearance_Hepatocyte_AZ"),
        "Clearance_Microsome_AZ": find_latest_results("primer_v2_Clearance_Microsome_AZ"),
    }

    print("📁 Found files:\n")
    print("Baseline (primer.py):")
    for dataset, file in baseline_files.items():
        print(f"  • {dataset}: {file if file else 'NOT FOUND'}")

    print("\nV2 (primer_v2.py):")
    for dataset, file in v2_files.items():
        print(f"  • {dataset}: {file if file else 'NOT FOUND'}")

    # Compare each dataset
    for dataset in baseline_files.keys():
        baseline_file = baseline_files[dataset]
        v2_file = v2_files[dataset]

        if not baseline_file:
            print(f"\n⚠️ {dataset}: No baseline results found - skipping")
            continue

        if not v2_file:
            print(f"\n⚠️ {dataset}: No V2 results found - skipping")
            continue

        print("\n" + "-"*80)
        print(f"  📊 {dataset}")
        print("-"*80)

        # Load data
        baseline_df = pd.read_csv(baseline_file)
        v2_df = pd.read_csv(v2_file)

        # Filter successful runs
        baseline_ok = baseline_df[baseline_df.get("success", True) == True]
        v2_ok = v2_df[v2_df.get("success", True) == True]

        if len(baseline_ok) == 0:
            print("  ⚠️ No successful baseline runs")
            continue

        if len(v2_ok) == 0:
            print("  ⚠️ No successful V2 runs")
            continue

        # Get best models
        baseline_best = baseline_ok.nsmallest(1, "val_rmse").iloc[0]
        v2_best = v2_ok.nsmallest(1, "val_rmse").iloc[0]

        # Stats
        print(f"\n  🏆 Best Models:")
        print(f"\n  Baseline (primer.py):")
        print(f"    Model: {baseline_best['model_name']}")
        print(f"    Layers: {int(baseline_best['graph_layers'])}, Hidden: {int(baseline_best['graph_hidden_channels'])}")
        print(f"    Val RMSE: {baseline_best['val_rmse']:.4f}")
        print(f"    Test RMSE: {baseline_best['test_rmse']:.4f}")
        print(f"    Test R²: {baseline_best['test_r2']:.4f}")

        print(f"\n  V2 (primer_v2.py):")
        print(f"    Model: {v2_best['model_name']}")
        print(f"    Layers: {int(v2_best['graph_layers'])}, Hidden: {int(v2_best['graph_hidden_channels'])}")
        print(f"    Val RMSE: {v2_best['val_rmse']:.4f}")
        print(f"    Test RMSE: {v2_best['test_rmse']:.4f}")
        print(f"    Test R²: {v2_best['test_r2']:.4f}")

        # Calculate improvements
        rmse_improvement = ((baseline_best['test_rmse'] - v2_best['test_rmse']) / baseline_best['test_rmse']) * 100
        r2_improvement = v2_best['test_r2'] - baseline_best['test_r2']

        print(f"\n  🎯 Improvements:")
        print(f"    RMSE: {rmse_improvement:+.2f}% {'✅' if rmse_improvement > 0 else '❌'}")
        print(f"    R²: {r2_improvement:+.4f} {'✅' if r2_improvement > 0 else '❌'}")

        # Variance comparison
        baseline_std = baseline_ok['test_rmse'].std()
        v2_std = v2_ok['test_rmse'].std()
        variance_reduction = ((baseline_std - v2_std) / baseline_std) * 100

        print(f"\n  📊 Stability:")
        print(f"    Baseline CV: {baseline_std/baseline_ok['test_rmse'].mean():.4f}")
        print(f"    V2 CV: {v2_std/v2_ok['test_rmse'].mean():.4f}")
        print(f"    Variance reduction: {variance_reduction:+.2f}% {'✅' if variance_reduction > 0 else '❌'}")

        # Top-5 comparison
        print(f"\n  📋 Top-5 Models:")
        print(f"\n    Baseline:")
        baseline_top5 = baseline_ok.nsmallest(5, "val_rmse")
        for i, (_, row) in enumerate(baseline_top5.iterrows(), 1):
            print(f"      {i}. {row['model_name']:8s} | RMSE: {row['test_rmse']:.4f} | R²: {row['test_r2']:.4f}")

        print(f"\n    V2:")
        v2_top5 = v2_ok.nsmallest(5, "val_rmse")
        for i, (_, row) in enumerate(v2_top5.iterrows(), 1):
            print(f"      {i}. {row['model_name']:8s} | RMSE: {row['test_rmse']:.4f} | R²: {row['test_r2']:.4f}")

    print("\n" + "="*80)
    print("  📈 Summary")
    print("="*80)
    print("\n  If you see improvements:")
    print("    ✅ Use primer_v2.py for production")
    print("    ✅ Consider enabling ensemble for critical predictions")
    print("\n  Next steps:")
    print("    1. Run full benchmark with more configs")
    print("    2. Test edge features impact")
    print("    3. Run 3-seed ensemble on best configs")
    print()

if __name__ == "__main__":
    compare_results()
