#!/usr/bin/env python3
"""
Batch runner for foundation model HPO experiments.

Runs HPO for multiple model+dataset+algorithm combinations based on config file.
"""
import argparse
import sys
import os
import yaml
from pathlib import Path
from itertools import product

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from scripts.run_foundation_hpo import run_foundation_hpo


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_batch_hpo(config_path: str = "config_foundation_benchmark.yaml", dry_run: bool = False):
    """
    Run batch HPO experiments from config file.
    
    Args:
        config_path: Path to config file
        dry_run: If True, only print what would be run
    """
    config = load_config(config_path)
    
    models = config.get('models', [])
    datasets = config.get('datasets', [])
    hpo_config = config.get('hpo', {})
    
    trials = hpo_config.get('trials', 30)
    max_iter = hpo_config.get('max_iter', 500)
    algorithms = hpo_config.get('algorithms', ['random'])
    
    device = config.get('device', 'auto')
    seed = config.get('seed', 42)
    output_dir = config.get('output_dir', 'runs/foundation')
    
    # Generate all combinations
    experiments = list(product(models, datasets, algorithms))
    
    print(f"\n{'='*80}")
    print(f"BATCH FOUNDATION MODEL HPO")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Total experiments: {len(experiments)}")
    print(f"  Models: {len(models)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Trials per experiment: {trials}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("DRY RUN - Would execute the following experiments:\n")
        for i, (model, dataset, algo) in enumerate(experiments, 1):
            print(f"{i:3d}. {model:12s} | {dataset:30s} | {algo:10s}")
        print(f"\nTotal: {len(experiments)} experiments")
        return
    
    # Run experiments
    results = []
    failed = []
    
    for i, (model, dataset, algo) in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*80}")
        
        try:
            output_path = run_foundation_hpo(
                model_type=model,
                dataset=dataset,
                algo_name=algo,
                trials=trials,
                max_iter=max_iter,
                seed=seed,
                device=device,
                output_dir=output_dir,
                verbose=True,
            )
            results.append({
                'model': model,
                'dataset': dataset,
                'algorithm': algo,
                'output': output_path,
                'status': 'success',
            })
            print(f"\n✓ Completed: {model} | {dataset} | {algo}")
        
        except KeyboardInterrupt:
            print("\n\nBatch interrupted by user")
            break
        
        except Exception as e:
            print(f"\n✗ Failed: {model} | {dataset} | {algo}")
            print(f"Error: {e}")
            failed.append({
                'model': model,
                'dataset': dataset,
                'algorithm': algo,
                'error': str(e),
                'status': 'failed',
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH HPO SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed experiments:")
        for f in failed:
            print(f"  - {f['model']} | {f['dataset']} | {f['algorithm']}: {f['error']}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch foundation model HPO from config file"
    )
    parser.add_argument("--config", type=str, default="config_foundation_benchmark.yaml",
                       help="Path to config file (default: config_foundation_benchmark.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be run without executing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        run_batch_hpo(config_path=args.config, dry_run=args.dry_run)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
