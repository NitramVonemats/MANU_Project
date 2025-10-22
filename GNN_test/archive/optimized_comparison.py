"""
Optimized comparison test: primer.py vs primer_v2.py
Target: Complete in ~1 hour
"""
import time
from datetime import datetime
import pandas as pd

print("="*80)
print("OPTIMIZED COMPARISON TEST")
print("primer.py (baseline) vs primer_v2.py (improved)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Strategy: 5 configs per dataset, 50 epochs (balanced speed vs accuracy)
# 5 configs × 3 datasets × 2 versions = 30 configs
# Estimated time: ~60 minutes

print("Configuration:")
print("  Configs per dataset: 5")
print("  Epochs: 50")
print("  Patience: 12")
print("  Expected time: ~60 minutes")
print()

all_results = {}

# Test 1: Baseline primer.py
print("\n" + "="*80)
print("TEST 1: Running primer.py (baseline)")
print("="*80)
t0 = time.time()

try:
    import primer

    baseline_results = primer.run_enhanced_molecular_benchmark(
        max_combos_per_dataset=5,
        epochs=50,
        patience=12,
        device='cpu',
        seed=42,
        out_prefix='optimized_baseline'
    )

    t1 = time.time()
    print(f"\nBaseline completed in {(t1-t0)/60:.1f} minutes")
    all_results['baseline'] = baseline_results

except Exception as e:
    print(f"\nERROR in baseline: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Improved primer_v2.py
print("\n" + "="*80)
print("TEST 2: Running primer_v2.py (EMA + Warmup + GatedAttn)")
print("="*80)
t0 = time.time()

try:
    import primer_v2

    v2_results = primer_v2.run_enhanced_molecular_benchmark(
        max_combos_per_dataset=5,
        epochs=50,
        patience=12,
        device='cpu',
        seed=42,
        out_prefix='optimized_v2',
        use_ensemble=False
    )

    t1 = time.time()
    print(f"\nV2 completed in {(t1-t0)/60:.1f} minutes")
    all_results['v2'] = v2_results

except Exception as e:
    print(f"\nERROR in v2: {e}")
    import traceback
    traceback.print_exc()

# Generate comparison report
print("\n" + "="*80)
print("COMPARISON REPORT")
print("="*80)

if 'baseline' in all_results and 'v2' in all_results:
    baseline_df = all_results['baseline']
    v2_df = all_results['v2']

    datasets = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]

    for dataset in datasets:
        print(f"\n{dataset}:")
        print("-" * 60)

        baseline_data = baseline_df[baseline_df['dataset'] == dataset]
        v2_data = v2_df[v2_df['dataset'] == dataset]

        if len(baseline_data) > 0 and len(v2_data) > 0:
            # Get best models
            baseline_best = baseline_data.nsmallest(1, 'val_rmse').iloc[0]
            v2_best = v2_data.nsmallest(1, 'val_rmse').iloc[0]

            # Calculate improvements
            rmse_improvement = ((baseline_best['test_rmse'] - v2_best['test_rmse']) / baseline_best['test_rmse']) * 100
            r2_improvement = v2_best['test_r2'] - baseline_best['test_r2']

            print(f"Baseline: {baseline_best['model_name']} | RMSE={baseline_best['test_rmse']:.3f}, R2={baseline_best['test_r2']:.3f}")
            print(f"V2:       {v2_best['model_name']} | RMSE={v2_best['test_rmse']:.3f}, R2={v2_best['test_r2']:.3f}")
            print(f"\nImprovement: RMSE {rmse_improvement:+.1f}%, R2 {r2_improvement:+.3f}")

            if rmse_improvement > 0:
                print("  -> V2 is BETTER")
            else:
                print("  -> Baseline is better")

print(f"\n{'='*80}")
print("TEST COMPLETED!")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
