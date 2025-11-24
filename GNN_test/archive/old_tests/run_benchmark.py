"""
PHASE 2: FOUNDATION MODEL BENCHMARKING
=======================================
Main script for running comprehensive benchmarks comparing:
1. GNN-only (Phase 1 baseline)
2. ChemBERTa-only (pretrained transformer)
3. GNN + ChemBERTa (hybrid fusion)

Usage:
    python run_benchmark.py                          # Run all datasets with all seeds
    python run_benchmark.py --dataset Half_Life_Obach  # Run specific dataset
    python run_benchmark.py --seeds 42 123           # Custom seeds
"""
import argparse
import pandas as pd
from datetime import datetime
from services.benchmark import benchmark_dataset

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

try:
    from rdkit import Chem
    RDKit_OK = True
except ImportError:
    RDKit_OK = False


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Foundation Model Benchmarking")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["Half_Life_Obach", "Clearance_Microsome_AZ", "Clearance_Hepatocyte_AZ", "ALL"],
        help="Specific dataset to benchmark (default: ALL)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for experiments (default: 42 123 456)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'#'*80}")
    print("# PHASE 2: FOUNDATION MODEL BENCHMARKING")
    print(f"{'#'*80}")
    print(f"\nTransformers available: {TRANSFORMERS_OK}")
    print(f"RDKit available: {RDKit_OK}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device or 'auto-detect'}")

    # Determine datasets to benchmark
    if args.dataset and args.dataset != "ALL":
        datasets = [args.dataset]
    else:
        datasets = ["Half_Life_Obach", "Clearance_Microsome_AZ", "Clearance_Hepatocyte_AZ"]

    print(f"\nDatasets: {', '.join(datasets)}")
    print(f"Estimated experiments: ~{len(datasets) * len(args.seeds) * 3} runs\n")

    # Run benchmarks
    all_results = []
    for dataset in datasets:
        results = benchmark_dataset(dataset, seeds=args.seeds, device=args.device)
        all_results.extend(results)

    # Save detailed results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = f"{args.output_dir}/phase2_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nDetailed results saved to: {csv_path}")

    # Generate summary
    summary_path = f"{args.output_dir}/phase2_summary_{timestamp}.csv"
    summary_data = []

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        for model_name in sorted(df_dataset['model_name'].unique()):
            df_model = df_dataset[df_dataset['model_name'] == model_name]
            summary_data.append({
                'dataset': dataset,
                'model': model_name,
                'mean_test_r2': df_model['test_r2'].mean(),
                'std_test_r2': df_model['test_r2'].std(),
                'mean_test_rmse': df_model['test_rmse'].mean(),
                'std_test_rmse': df_model['test_rmse'].std(),
                'mean_test_spearman': df_model['test_spearman'].mean(),
                'n_seeds': len(df_model)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY BY MODEL TYPE:")
    print(f"{'='*80}")

    for dataset in datasets:
        print(f"\n{dataset}:")
        df_dataset = df[df['dataset'] == dataset]
        for model_name in sorted(df_dataset['model_name'].unique()):
            df_model = df_dataset[df_dataset['model_name'] == model_name]
            mean_r2 = df_model['test_r2'].mean()
            std_r2 = df_model['test_r2'].std()
            mean_rmse = df_model['test_rmse'].mean()
            print(f"  {model_name:20} R2={mean_r2:6.3f} ± {std_r2:.3f}  RMSE={mean_rmse:6.2f}")

    print(f"\n{'='*80}")
    print("Phase 2 Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
