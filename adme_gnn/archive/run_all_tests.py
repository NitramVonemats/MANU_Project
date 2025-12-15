"""
Comprehensive Testing Suite for primer.py vs primer_v2.py
Runs all tests and generates detailed comparison report
"""
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Test configurations
TEST_CONFIGS = {
    "quick_test": {
        "max_combos": 5,
        "epochs": 50,
        "patience": 10,
        "description": "Quick sanity check (5 configs, 50 epochs)"
    },
    "standard_test": {
        "max_combos": 10,
        "epochs": 80,
        "patience": 15,
        "description": "Standard test (10 configs, 80 epochs)"
    },
    "full_test": {
        "max_combos": 20,
        "epochs": 120,
        "patience": 25,
        "description": "Full benchmark (20 configs, 120 epochs)"
    }
}

DATASETS = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80)

def run_primer_baseline(test_config, device="cuda"):
    """Test 1: Run baseline primer.py"""
    print_section("TEST 1: Baseline primer.py Performance")

    try:
        import primer as primer_old

        results = primer_old.run_enhanced_molecular_benchmark(
            max_combos_per_dataset=test_config["max_combos"],
            epochs=test_config["epochs"],
            patience=test_config["patience"],
            device=device,
            seed=42,
            out_prefix="test_primer_baseline"
        )

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_primer_v2_single_seed(test_config, device="cuda"):
    """Test 2: Run primer_v2.py with single seed"""
    print_section("TEST 2: primer_v2.py with Single Seed (EMA + Warmup + GatedAttn)")

    try:
        import primer_v2

        results = primer_v2.run_enhanced_molecular_benchmark(
            max_combos_per_dataset=test_config["max_combos"],
            epochs=test_config["epochs"],
            patience=test_config["patience"],
            device=device,
            seed=42,
            out_prefix="test_primer_v2_single",
            use_ensemble=False
        )

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_primer_v2_edge_features(test_config, device="cuda"):
    """Test 3: Test edge features impact"""
    print_section("TEST 3: Edge Features Impact Test")

    try:
        import primer_v2

        # Modify tester to enable edge features
        original_create = primer_v2.EnhancedGNNTester.create_enhanced_parameter_combinations

        def create_with_edges(self, max_combinations=50):
            combos = original_create(self, max_combinations)
            # Force enable edge features for half the configs
            for i, cfg in enumerate(combos):
                if i % 2 == 0:
                    cfg["use_edge_features"] = True
            return combos

        primer_v2.EnhancedGNNTester.create_enhanced_parameter_combinations = create_with_edges

        results = primer_v2.run_enhanced_molecular_benchmark(
            max_combos_per_dataset=test_config["max_combos"],
            epochs=test_config["epochs"],
            patience=test_config["patience"],
            device=device,
            seed=42,
            out_prefix="test_primer_v2_edges",
            use_ensemble=False
        )

        # Restore original
        primer_v2.EnhancedGNNTester.create_enhanced_parameter_combinations = original_create

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_seed_ensemble_test(device="cuda"):
    """Test 4: 3-seed ensemble on best configs"""
    print_section("TEST 4: Seed Ensemble Test (3 seeds, median)")

    try:
        import primer_v2

        # Run smaller test with ensemble
        results = primer_v2.run_enhanced_molecular_benchmark(
            max_combos_per_dataset=5,  # Only 5 configs for ensemble
            epochs=80,
            patience=15,
            device=device,
            seed=42,
            out_prefix="test_ensemble",
            use_ensemble=True
        )

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def analyze_results(baseline_results, v2_results, edge_results=None, ensemble_results=None):
    """Generate comprehensive comparison report"""
    print_header("COMPREHENSIVE ANALYSIS REPORT")

    report = {
        "timestamp": datetime.now().isoformat(),
        "comparisons": {},
        "improvements": {},
        "recommendations": []
    }

    # Compare baseline vs v2
    print_section("1. Baseline vs V2 Comparison")

    for dataset in DATASETS:
        print(f"\n  Dataset: {dataset}")

        # Filter results for this dataset
        baseline_df = baseline_results["results"]
        v2_df = v2_results["results"]

        baseline_data = baseline_df[baseline_df["dataset"] == dataset]
        v2_data = v2_df[v2_df["dataset"] == dataset]

        if len(baseline_data) == 0 or len(v2_data) == 0:
            print("    No data available")
            continue

        # Get best results
        baseline_best = baseline_data.nsmallest(1, "val_rmse").iloc[0]
        v2_best = v2_data.nsmallest(1, "val_rmse").iloc[0]

        # Calculate improvements
        rmse_improvement = ((baseline_best["test_rmse"] - v2_best["test_rmse"]) / baseline_best["test_rmse"]) * 100
        r2_improvement = v2_best["test_r2"] - baseline_best["test_r2"]

        print(f"\n  Baseline Best:")
        print(f"    Model: {baseline_best['model_name']}")
        print(f"    Test RMSE: {baseline_best['test_rmse']:.4f}")
        print(f"    Test R : {baseline_best['test_r2']:.4f}")

        print(f"\n  V2 Best:")
        print(f"    Model: {v2_best['model_name']}")
        print(f"    Test RMSE: {v2_best['test_rmse']:.4f}")
        print(f"    Test R : {v2_best['test_r2']:.4f}")

        print(f"\n    Improvements:")
        print(f"    RMSE: {rmse_improvement:+.2f}% {' ' if rmse_improvement > 0 else ' '}")
        print(f"    R : {r2_improvement:+.4f} {' ' if r2_improvement > 0 else ' '}")

        report["comparisons"][dataset] = {
            "baseline_rmse": float(baseline_best["test_rmse"]),
            "v2_rmse": float(v2_best["test_rmse"]),
            "rmse_improvement_pct": float(rmse_improvement),
            "baseline_r2": float(baseline_best["test_r2"]),
            "v2_r2": float(v2_best["test_r2"]),
            "r2_improvement": float(r2_improvement)
        }

        # Calculate variance reduction
        baseline_std = baseline_data["test_rmse"].std()
        v2_std = v2_data["test_rmse"].std()
        variance_reduction = ((baseline_std - v2_std) / baseline_std) * 100

        print(f"    Variance: {variance_reduction:+.2f}% {'  (more stable)' if variance_reduction > 0 else ' '}")

        report["comparisons"][dataset]["variance_reduction_pct"] = float(variance_reduction)

    # Edge features analysis
    if edge_results and edge_results["status"] == "success":
        print_section("2. Edge Features Impact")

        edge_df = edge_results["results"]

        for dataset in DATASETS:
            edge_data = edge_df[edge_df["dataset"] == dataset]

            if len(edge_data) == 0:
                continue

            with_edges = edge_data[edge_data.get("use_edge_features", False) == True]
            without_edges = edge_data[edge_data.get("use_edge_features", False) == False]

            if len(with_edges) > 0 and len(without_edges) > 0:
                avg_rmse_with = with_edges["test_rmse"].mean()
                avg_rmse_without = without_edges["test_rmse"].mean()

                edge_impact = ((avg_rmse_without - avg_rmse_with) / avg_rmse_without) * 100

                print(f"\n  {dataset}:")
                print(f"  Avg RMSE with edges: {avg_rmse_with:.4f}")
                print(f"  Avg RMSE without edges: {avg_rmse_without:.4f}")
                print(f"  Impact: {edge_impact:+.2f}% {'  (helpful)' if edge_impact > 0 else '  (harmful)'}")

                if edge_impact > 0:
                    report["recommendations"].append(f"Enable edge features for {dataset}")
                else:
                    report["recommendations"].append(f"Keep edge features disabled for {dataset}")

    # Ensemble analysis
    if ensemble_results and ensemble_results["status"] == "success":
        print_section("3. Ensemble Performance")

        ensemble_df = ensemble_results["results"]

        for dataset in DATASETS:
            ensemble_data = ensemble_df[ensemble_df["dataset"] == dataset]

            if len(ensemble_data) > 0 and "val_rmse_std" in ensemble_data.columns:
                best_ensemble = ensemble_data.nsmallest(1, "val_rmse").iloc[0]

                print(f"\n  {dataset}:")
                print(f"  Median RMSE: {best_ensemble.get('test_rmse', 'N/A'):.4f}")
                print(f"  Std Dev: {best_ensemble.get('test_rmse_std', 'N/A'):.4f}")
                print(f"  Median R : {best_ensemble.get('test_r2', 'N/A'):.4f}")

                report["recommendations"].append(f"Ensemble reduces variance for {dataset}")

    # Overall summary
    print_section("4. Overall Summary")

    total_improvements = sum(1 for comp in report["comparisons"].values() if comp["rmse_improvement_pct"] > 0)
    total_datasets = len(report["comparisons"])

    print(f"\n  Improvements on {total_improvements}/{total_datasets} datasets")

    avg_rmse_improvement = np.mean([comp["rmse_improvement_pct"] for comp in report["comparisons"].values()])
    avg_r2_improvement = np.mean([comp["r2_improvement"] for comp in report["comparisons"].values()])

    print(f"  Average RMSE improvement: {avg_rmse_improvement:+.2f}%")
    print(f"  Average R  improvement: {avg_r2_improvement:+.4f}")

    report["summary"] = {
        "success_rate": total_improvements / total_datasets,
        "avg_rmse_improvement_pct": float(avg_rmse_improvement),
        "avg_r2_improvement": float(avg_r2_improvement)
    }

    # Recommendations
    print_section("5. Recommendations")

    if avg_rmse_improvement > 10:
        print("\n  primer_v2.py shows significant improvements (>10% RMSE reduction)")
        print("     Use primer_v2.py for production")
        report["recommendations"].append("Use primer_v2 for production")
    elif avg_rmse_improvement > 5:
        print("\n  primer_v2.py shows moderate improvements (5-10% RMSE reduction)")
        print("     Consider using primer_v2.py with ensemble for critical tasks")
        report["recommendations"].append("Use primer_v2 with ensemble for critical tasks")
    else:
        print("\n  primer_v2.py shows minimal improvements (<5% RMSE reduction)")
        print("     May need further tuning or dataset-specific adjustments")
        report["recommendations"].append("Further tuning needed")

    if report["recommendations"]:
        print("\n  Action Items:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

    return report

def save_report(report, filename="test_report.json"):
    """Save report to JSON"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {filename}")

def main():
    print_header("COMPREHENSIVE TESTING SUITE")
    print("Testing primer.py vs primer_v2.py")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Select test configuration
    print("\nAvailable test configurations:")
    for i, (key, config) in enumerate(TEST_CONFIGS.items(), 1):
        print(f"  {i}. {key}: {config['description']}")

    # Use standard test by default
    test_config = TEST_CONFIGS["standard_test"]
    print(f"\nUsing: {test_config['description']}")

    # Check device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
    except:
        device = "cpu"
        print(f"Device: {device} (PyTorch not detected)")

    # Run tests
    all_results = {}

    # Test 1: Baseline
    print_header("Running Test 1: Baseline primer.py")
    t0 = time.time()
    baseline_results = run_primer_baseline(test_config, device)
    t1 = time.time()
    print(f"  Completed in {(t1-t0)/60:.1f} minutes")
    all_results["baseline"] = baseline_results

    # Test 2: V2 single seed
    print_header("Running Test 2: primer_v2.py (single seed)")
    t0 = time.time()
    v2_results = run_primer_v2_single_seed(test_config, device)
    t1 = time.time()
    print(f"  Completed in {(t1-t0)/60:.1f} minutes")
    all_results["v2_single"] = v2_results

    # Test 3: Edge features (optional, only if time permits)
    print_header("Running Test 3: Edge Features Test")
    print("  This test is optional and takes extra time.")
    print("Skipping for now - can be run separately if needed.")
    edge_results = None
    # Uncomment to enable:
    # t0 = time.time()
    # edge_results = run_primer_v2_edge_features(test_config, device)
    # t1 = time.time()
    # print(f"  Completed in {(t1-t0)/60:.1f} minutes")
    # all_results["edge_test"] = edge_results

    # Test 4: Ensemble (optional, only for final validation)
    print_header("Running Test 4: Seed Ensemble Test")
    print("  This test runs 3x longer (3 seeds per config).")
    print("Skipping for now - run separately for final validation.")
    ensemble_results = None
    # Uncomment to enable:
    # t0 = time.time()
    # ensemble_results = run_seed_ensemble_test(device)
    # t1 = time.time()
    # print(f"  Completed in {(t1-t0)/60:.1f} minutes")
    # all_results["ensemble"] = ensemble_results

    # Generate report
    if baseline_results["status"] == "success" and v2_results["status"] == "success":
        report = analyze_results(baseline_results, v2_results, edge_results, ensemble_results)
        save_report(report, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        print("\n  Tests failed, cannot generate report")
        if baseline_results["status"] == "failed":
            print(f"   Baseline error: {baseline_results['error']}")
        if v2_results["status"] == "failed":
            print(f"   V2 error: {v2_results['error']}")

    print_header("  ALL TESTS COMPLETED")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
