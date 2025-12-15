#!/usr/bin/env python3
"""
Benchmark Report Generator
Generates comprehensive tables and plots comparing performance across datasets and models.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Matplotlib setup - handle non-GUI backends
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_hpo_results(runs_dir: str = "runs") -> pd.DataFrame:
    """Load all HPO results from JSON files."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return pd.DataFrame()
    
    results = []
    
    for dataset_dir in runs_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        json_files = list(dataset_dir.glob("hpo_*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                algo_name = data.get("algo", "unknown")
                search = data.get("search", {})
                final = data.get("final_training", {})
                
                results.append({
                    "dataset": dataset_name,
                    "model_type": f"HPO-{algo_name}",
                    "algorithm": algo_name,
                    "trials": search.get("trials", 0),
                    "best_val_rmse_search": search.get("best_val_rmse", None),
                    "test_rmse": final.get("test_metrics", {}).get("rmse", None),
                    "test_mae": final.get("test_metrics", {}).get("mae", None),
                    "test_r2": final.get("test_metrics", {}).get("r2", None),
                    "val_rmse": final.get("val_metrics", {}).get("rmse", None),
                    "val_mae": final.get("val_metrics", {}).get("mae", None),
                    "val_r2": final.get("val_metrics", {}).get("r2", None),
                    "train_time": final.get("train_time", None),
                    "epochs": final.get("epochs", None),
                    "patience": final.get("patience", None),
                    "hidden_dim": search.get("best_params", {}).get("hidden_dim", None),
                    "num_layers": search.get("best_params", {}).get("num_layers", None),
                    "lr": search.get("best_params", {}).get("lr", None),
                    "weight_decay": search.get("best_params", {}).get("weight_decay", None),
                    "file_path": str(json_file),
                })
            except Exception as e:
                print(f"WARNING: Error loading {json_file}: {e}")
                continue
    
    return pd.DataFrame(results)


def load_basic_results(basic_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Load basic training results from run_all_hpo.py results."""
    if not basic_results:
        return pd.DataFrame()
    
    results = []
    
    for result in basic_results:
        if result.get("result") is None:
            continue
        
        dataset = result.get("dataset", "unknown")
        res_data = result.get("result", {})
        test_metrics = res_data.get("test_metrics", {})
        val_metrics = res_data.get("val_metrics", {})
        
        results.append({
            "dataset": dataset,
            "model_type": "Basic",
            "algorithm": "basic",
            "trials": 1,
            "best_val_rmse_search": None,
            "test_rmse": test_metrics.get("rmse", None),
            "test_mae": test_metrics.get("mae", None),
            "test_r2": test_metrics.get("r2", None),
            "val_rmse": val_metrics.get("rmse", None),
            "val_mae": val_metrics.get("mae", None),
            "val_r2": val_metrics.get("r2", None),
            "train_time": res_data.get("train_time", None),
            "epochs": None,
            "patience": None,
            "hidden_dim": None,
            "num_layers": None,
            "lr": None,
            "weight_decay": None,
            "file_path": None,
        })
    
    return pd.DataFrame(results)


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table comparing models across datasets."""
    if df.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        
        # Best HPO result
        hpo_df = dataset_df[dataset_df["model_type"] != "Basic"]
        if not hpo_df.empty:
            best_hpo = hpo_df.loc[hpo_df["test_rmse"].idxmin()]
            summary_data.append({
                "dataset": dataset,
                "model": "Best HPO",
                "algorithm": best_hpo["algorithm"],
                "test_rmse": best_hpo["test_rmse"],
                "test_mae": best_hpo["test_mae"],
                "test_r2": best_hpo["test_r2"],
                "val_rmse": best_hpo["val_rmse"],
                "train_time": best_hpo["train_time"],
            })
        
        # Basic model result
        basic_df = dataset_df[dataset_df["model_type"] == "Basic"]
        if not basic_df.empty:
            basic = basic_df.iloc[0]
            summary_data.append({
                "dataset": dataset,
                "model": "Basic",
                "algorithm": "basic",
                "test_rmse": basic["test_rmse"],
                "test_mae": basic["test_mae"],
                "test_r2": basic["test_r2"],
                "val_rmse": basic["val_rmse"],
                "train_time": basic["train_time"],
            })
    
    return pd.DataFrame(summary_data)


def plot_performance_comparison(df: pd.DataFrame, output_dir: str):
    """Create performance comparison plots."""
    if df.empty:
        print("WARNING: No data to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Test RMSE comparison across datasets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Performance Comparison: HPO vs Basic Models", fontsize=16, fontweight='bold')
    
    # Test RMSE
    ax = axes[0, 0]
    summary_df = create_summary_table(df)
    if not summary_df.empty:
        pivot_rmse = summary_df.pivot(index="dataset", columns="model", values="test_rmse")
        pivot_rmse.plot(kind="bar", ax=ax, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_title("Test RMSE Comparison", fontweight='bold')
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Dataset")
        ax.legend(title="Model Type")
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    # Test R²
    ax = axes[0, 1]
    if not summary_df.empty:
        pivot_r2 = summary_df.pivot(index="dataset", columns="model", values="test_r2")
        pivot_r2.plot(kind="bar", ax=ax, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_title("Test R² Comparison", fontweight='bold')
        ax.set_ylabel("R²")
        ax.set_xlabel("Dataset")
        ax.legend(title="Model Type")
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    # Test MAE
    ax = axes[1, 0]
    if not summary_df.empty:
        pivot_mae = summary_df.pivot(index="dataset", columns="model", values="test_mae")
        pivot_mae.plot(kind="bar", ax=ax, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_title("Test MAE Comparison", fontweight='bold')
        ax.set_ylabel("MAE")
        ax.set_xlabel("Dataset")
        ax.legend(title="Model Type")
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    # Training time comparison
    ax = axes[1, 1]
    if not summary_df.empty:
        pivot_time = summary_df.pivot(index="dataset", columns="model", values="train_time")
        pivot_time.plot(kind="bar", ax=ax, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_title("Training Time Comparison", fontweight='bold')
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("Dataset")
        ax.legend(title="Model Type")
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_dir}/performance_comparison.png")
    
    # 2. Dataset-specific performance
    datasets = df["dataset"].unique()
    if len(datasets) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Performance by Dataset", fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, dataset in enumerate(datasets[:4]):  # Max 4 datasets
            ax = axes[idx]
            dataset_df = df[df["dataset"] == dataset]
            
            if not dataset_df.empty:
                x_pos = np.arange(len(dataset_df))
                width = 0.35
                
                test_rmse = dataset_df["test_rmse"].values
                val_rmse = dataset_df["val_rmse"].values
                model_types = dataset_df["model_type"].values
                
                bars1 = ax.bar(x_pos - width/2, test_rmse, width, label='Test RMSE', alpha=0.8, color='#3498db')
                bars2 = ax.bar(x_pos + width/2, val_rmse, width, label='Val RMSE', alpha=0.8, color='#e74c3c')
                
                ax.set_title(f"{dataset}", fontweight='bold')
                ax.set_ylabel("RMSE")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_types, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(datasets), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dataset_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_dir}/dataset_performance.png")
    
    # 3. Algorithm comparison (if multiple HPO algorithms)
    hpo_df = df[df["model_type"] != "Basic"]
    if len(hpo_df) > 0 and hpo_df["algorithm"].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("HPO Algorithm Comparison", fontsize=16, fontweight='bold')
        
        # RMSE by algorithm
        ax = axes[0]
        algo_rmse = hpo_df.groupby("algorithm")["test_rmse"].mean().sort_values()
        algo_rmse.plot(kind="barh", ax=ax, color='#9b59b6', alpha=0.8)
        ax.set_title("Average Test RMSE by Algorithm", fontweight='bold')
        ax.set_xlabel("Test RMSE")
        ax.grid(axis='x', alpha=0.3)
        
        # R² by algorithm
        ax = axes[1]
        algo_r2 = hpo_df.groupby("algorithm")["test_r2"].mean().sort_values(ascending=False)
        algo_r2.plot(kind="barh", ax=ax, color='#2ecc71', alpha=0.8)
        ax.set_title("Average Test R² by Algorithm", fontweight='bold')
        ax.set_xlabel("Test R²")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_dir}/algorithm_comparison.png")


def generate_benchmark_report(
    basic_results: Optional[List[Dict[str, Any]]] = None,
    runs_dir: str = "runs",
    output_dir: str = "reports"
) -> str:
    """Generate comprehensive benchmark report."""
    print(f"\n{'='*80}")
    print("GENERATING BENCHMARK REPORT")
    print(f"{'='*80}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(output_dir, f"benchmark_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    hpo_df = load_hpo_results(runs_dir)
    basic_df = load_basic_results(basic_results or [])

    if hpo_df.empty and basic_df.empty:
        print("WARNING: No results found to generate report")
        return report_dir
    
    # Combine results
    df = pd.concat([hpo_df, basic_df], ignore_index=True) if not basic_df.empty else hpo_df
    
    print(f"   Loaded {len(hpo_df)} HPO results")
    print(f"   Loaded {len(basic_df)} basic training results")
    print(f"   Total: {len(df)} results\n")
    
    # Generate summary table
    print("Generating summary tables...")
    summary_df = create_summary_table(df)

    # Save full results
    full_csv = os.path.join(report_dir, "full_results.csv")
    df.to_csv(full_csv, index=False)
    print(f"[OK] Saved: {full_csv}")

    # Save summary
    summary_csv = os.path.join(report_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Saved: {summary_csv}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No summary data available")
    
    # Generate plots
    print(f"\nGenerating plots...")
    plot_performance_comparison(df, report_dir)

    # Generate detailed comparison table
    print(f"\nGenerating detailed comparison table...")
    if not df.empty:
        comparison_cols = [
            "dataset", "model_type", "algorithm", "test_rmse", "test_mae", "test_r2",
            "val_rmse", "train_time"
        ]
        available_cols = [col for col in comparison_cols if col in df.columns]
        comparison_df = df[available_cols].copy()
        comparison_df = comparison_df.sort_values(["dataset", "test_rmse"])
        
        comparison_csv = os.path.join(report_dir, "detailed_comparison.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"[OK] Saved: {comparison_csv}")

        # Print comparison
        print(f"\n{'='*80}")
        print("DETAILED COMPARISON")
        print(f"{'='*80}\n")
        print(comparison_df.to_string(index=False))
    
    # Generate markdown report
    print(f"\nGenerating markdown report...")
    md_path = os.path.join(report_dir, "README.md")
    with open(md_path, 'w') as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        if not summary_df.empty:
            try:
                f.write(summary_df.to_markdown(index=False))
            except AttributeError:
                # Fallback if to_markdown is not available
                f.write(summary_df.to_string(index=False))
            f.write("\n\n")
        
        f.write("## Files\n\n")
        f.write("- `full_results.csv` - Complete results from all runs\n")
        f.write("- `summary.csv` - Summary comparison table\n")
        f.write("- `detailed_comparison.csv` - Detailed comparison sorted by performance\n")
        f.write("- `performance_comparison.png` - Performance comparison plots\n")
        f.write("- `dataset_performance.png` - Dataset-specific performance\n")
        if len(df[df["model_type"] != "Basic"]) > 0 and df["algorithm"].nunique() > 1:
            f.write("- `algorithm_comparison.png` - HPO algorithm comparison\n")
    
    print(f"[OK] Saved: {md_path}")

    print(f"\n{'='*80}")
    print(f"[OK] BENCHMARK REPORT GENERATED")
    print(f"{'='*80}")
    print(f"Report directory: {report_dir}")
    print(f"\nFiles generated:")
    print(f"  - full_results.csv")
    print(f"  - summary.csv")
    print(f"  - detailed_comparison.csv")
    print(f"  - README.md")
    print(f"  - performance_comparison.png")
    print(f"  - dataset_performance.png")
    if len(df[df["model_type"] != "Basic"]) > 0 and df["algorithm"].nunique() > 1:
        print(f"  - algorithm_comparison.png")
    
    return report_dir


if __name__ == "__main__":
    # Test the report generator
    report_dir = generate_benchmark_report()
    print(f"\n[OK] Test report generated in: {report_dir}")

