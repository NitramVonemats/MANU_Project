#!/usr/bin/env python3
"""
Automated runner for all datasets.
Runs hyperparameter optimization (HPO) and/or basic training for all datasets.

Usage:
    # Both HPO and basic training (default)
    python run_all_hpo.py
    python run_all_hpo.py --trials 10 --epochs 50
    
    # HPO only
    python run_all_hpo.py --mode hpo --trials 20 --epochs 100
    
    # Basic training only (no HPO)
    python run_all_hpo.py --mode basic --epochs 100
    python run_all_hpo.py --mode basic --datasets Half_Life_Obach Caco2_Wang
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# All available datasets
ALL_DATASETS = [
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "Caco2_Wang",
]

# Default HPO configuration
DEFAULT_TRIALS = 10
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 12
DEFAULT_ALGO = "random"  # Options: "random", "pso", "ga", "sa", "hc", "abc"
DEFAULT_BATCH_TRAIN = 32
DEFAULT_BATCH_EVAL = 64
DEFAULT_SEED = 42


def run_basic_training_for_dataset(
    dataset: str,
    epochs: int = 100,
    patience: int = 20,
    device: str = "auto",
    seed: int = DEFAULT_SEED,
):
    """Run basic training (no HPO) for a single dataset."""
    print(f"\n{'='*80}")
    print(f"🧬 Starting basic training for: {dataset}")
    print(f"   Epochs: {epochs}, Patience: {patience}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Import and run training directly
    try:
        from src.optimized_gnn import train_model
        
        result = train_model(
            dataset_name=dataset,
            epochs=epochs,
            patience=patience,
            device=device,
            seed=seed,
            verbose=True,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {dataset} in {elapsed:.1f}s")
        
        # Extract metrics from result
        test_rmse = result.get("test_metrics", {}).get("rmse", "N/A")
        val_rmse = result.get("val_metrics", {}).get("rmse", "N/A")
        print(f"   Test RMSE: {test_rmse:.6f}" if isinstance(test_rmse, float) else f"   Test RMSE: {test_rmse}")
        print(f"   Val RMSE:  {val_rmse:.6f}" if isinstance(val_rmse, float) else f"   Val RMSE:  {val_rmse}")
        
        return True, elapsed, result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed {dataset} after {elapsed:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed, None
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted during {dataset}")
        return None, time.time() - start_time, None


def run_hpo_for_dataset(
    dataset: str, 
    algo: str = "random",
    trials: int = DEFAULT_TRIALS,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    batch_train: int = DEFAULT_BATCH_TRAIN,
    batch_eval: int = DEFAULT_BATCH_EVAL,
    seed: int = DEFAULT_SEED,
):
    """Run HPO for a single dataset."""
    print(f"\n{'='*80}")
    print(f"🧬 Starting HPO for: {dataset}")
    print(f"   Algorithm: {algo}")
    print(f"   Trials: {trials}, Epochs: {epochs}, Patience: {patience}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable, "-m", f"hpo.{algo}",
        "--dataset", dataset,
        "--trials", str(trials),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch-train", str(batch_train),
        "--batch-eval", str(batch_eval),
        "--seed", str(seed),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
        )
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {dataset} in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed {dataset} after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False, elapsed
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted during {dataset}")
        return None, time.time() - start_time


def main():
    """Run HPO and/or basic training for all datasets."""
    parser = argparse.ArgumentParser(
        description="Run HPO or basic training for all datasets automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both HPO and basic training (default)
  python run_all_hpo.py
  python run_all_hpo.py --trials 10 --epochs 50
  
  # HPO only
  python run_all_hpo.py --mode hpo --trials 20 --epochs 100
  python run_all_hpo.py --mode hpo --datasets Half_Life_Obach Caco2_Wang --algo pso
  
  # Basic training only (no HPO)
  python run_all_hpo.py --mode basic --epochs 100
  python run_all_hpo.py --mode basic --datasets Half_Life_Obach Caco2_Wang --patience 20
  python run_all_hpo.py --mode basic --no-confirm
        """
    )
    parser.add_argument("--mode", type=str, default="both",
                       choices=["hpo", "basic", "both"],
                       help="Mode: 'hpo' for HPO only, 'basic' for basic training only, 'both' for both (default: both)")
    parser.add_argument("--datasets", nargs="+", default=None,
                       help=f"Datasets to run (default: all). Options: {', '.join(ALL_DATASETS)}")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                       help=f"Number of HPO trials per dataset (HPO mode only, default: {DEFAULT_TRIALS})")
    parser.add_argument("--epochs", type=int, default=None,
                       help=f"Number of training epochs (default: {DEFAULT_EPOCHS} for HPO, 100 for basic)")
    parser.add_argument("--patience", type=int, default=None,
                       help=f"Early stopping patience (default: {DEFAULT_PATIENCE} for HPO, 20 for basic)")
    parser.add_argument("--algo", type=str, default=DEFAULT_ALGO,
                       choices=["random", "pso", "ga", "sa", "hc", "abc"],
                       help=f"HPO algorithm (HPO mode only, default: {DEFAULT_ALGO})")
    parser.add_argument("--batch-train", type=int, default=DEFAULT_BATCH_TRAIN, dest="batch_train",
                       help=f"Training batch size (HPO mode only, default: {DEFAULT_BATCH_TRAIN})")
    parser.add_argument("--batch-eval", type=int, default=DEFAULT_BATCH_EVAL, dest="batch_eval",
                       help=f"Evaluation batch size (HPO mode only, default: {DEFAULT_BATCH_EVAL})")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (default: auto)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    # Set defaults based on mode
    run_hpo = args.mode in ["hpo", "both"]
    run_basic = args.mode in ["basic", "both"]
    
    # Set HPO defaults
    hpo_epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    hpo_patience = args.patience if args.patience is not None else DEFAULT_PATIENCE
    
    # Set basic training defaults
    basic_epochs = args.epochs if args.epochs is not None else 100
    basic_patience = args.patience if args.patience is not None else 20
    
    # Determine datasets to run
    datasets = args.datasets if args.datasets else ALL_DATASETS
    
    # Validate datasets
    invalid = [d for d in datasets if d not in ALL_DATASETS]
    if invalid:
        print(f"❌ Error: Invalid datasets: {invalid}")
        print(f"Valid options: {', '.join(ALL_DATASETS)}")
        sys.exit(1)
    
    # Print header based on mode
    if args.mode == "both":
        mode_title = "HPO + BASIC TRAINING RUNNER"
    elif args.mode == "hpo":
        mode_title = "HPO RUNNER"
    else:
        mode_title = "BASIC TRAINING RUNNER"
    
    print(f"\n{'#'*80}")
    print(f"# AUTOMATED {mode_title}")
    print(f"{'#'*80}")
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Datasets: {', '.join(datasets)}")
    
    if run_hpo:
        print(f"\n  HPO Settings:")
        print(f"    Algorithm: {args.algo}")
        print(f"    Trials per dataset: {args.trials}")
        print(f"    Epochs per trial: {hpo_epochs}")
        print(f"    Patience: {hpo_patience}")
        print(f"    Batch sizes: train={args.batch_train}, eval={args.batch_eval}")
    
    if run_basic:
        print(f"\n  Basic Training Settings:")
        print(f"    Epochs: {basic_epochs}")
        print(f"    Patience: {basic_patience}")
        print(f"    Device: {args.device}")
    
    print(f"\n  Seed: {args.seed}")
    
    # Calculate estimated time
    total_runs = 0
    if run_hpo:
        hpo_runs = len(datasets) * args.trials
        total_runs += hpo_runs
        print(f"\n  HPO runs: {hpo_runs}")
    if run_basic:
        basic_runs = len(datasets)
        total_runs += basic_runs
        print(f"\n  Basic training runs: {basic_runs}")
    
    est_minutes = (len(datasets) * args.trials * 2 if run_hpo else 0) + (len(datasets) * 5 if run_basic else 0)
    print(f"\n  Estimated time: ~{est_minutes} minutes (very rough estimate)")
    
    confirm_msg = f"This will run {args.mode.upper()} for all specified datasets. Continue? [y/N]: "
    
    if not args.no_confirm:
        response = input(f"\n⚠️  {confirm_msg}")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    total_start = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_results = []
    
    print(f"\n🚀 Starting at {start_datetime}")
    print("="*80)
    
    # Run HPO if needed
    if run_hpo:
        print(f"\n\n{'='*80}")
        print(f"PHASE 1: HPO for all datasets")
        print(f"{'='*80}")
        hpo_results = []
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n\n[HPO {i}/{len(datasets)}] Processing {dataset}...")
            
            success, elapsed = run_hpo_for_dataset(
                dataset, 
                args.algo,
                trials=args.trials,
                epochs=hpo_epochs,
                patience=hpo_patience,
                batch_train=args.batch_train,
                batch_eval=args.batch_eval,
                seed=args.seed,
            )
            
            hpo_results.append({
                "dataset": dataset,
                "mode": "hpo",
                "success": success,
                "time_seconds": elapsed,
            })
            all_results.append(hpo_results[-1])
            
            if success is None:  # Interrupted
                print(f"\n⚠️  Stopped by user. Processed {i-1}/{len(datasets)} datasets.")
                break
            
            # Brief pause between datasets
            if i < len(datasets):
                print("\n⏸️  Waiting 2 seconds before next dataset...")
                time.sleep(2)
        
        print(f"\n\n{'='*80}")
        print(f"✅ HPO phase completed")
        print(f"{'='*80}")
    
    # Run basic training if needed
    if run_basic:
        print(f"\n\n{'='*80}")
        print(f"PHASE 2: Basic training for all datasets")
        print(f"{'='*80}")
        basic_results = []
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n\n[Basic {i}/{len(datasets)}] Processing {dataset}...")
            
            success, elapsed, result_data = run_basic_training_for_dataset(
                dataset,
                epochs=basic_epochs,
                patience=basic_patience,
                device=args.device,
                seed=args.seed,
            )
            
            basic_results.append({
                "dataset": dataset,
                "mode": "basic",
                "success": success,
                "time_seconds": elapsed,
                "result": result_data,
            })
            all_results.append(basic_results[-1])
            
            if success is None:  # Interrupted
                print(f"\n⚠️  Stopped by user. Processed {i-1}/{len(datasets)} datasets.")
                break
            
            # Brief pause between datasets
            if i < len(datasets):
                print("\n⏸️  Waiting 2 seconds before next dataset...")
                time.sleep(2)
        
        print(f"\n\n{'='*80}")
        print(f"✅ Basic training phase completed")
        print(f"{'='*80}")
    
    total_elapsed = time.time() - total_start
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Summary
    print(f"\n\n{'#'*80}")
    print("# FINAL SUMMARY")
    print(f"{'#'*80}")
    print(f"\nStarted: {start_datetime}")
    print(f"Ended:   {end_datetime}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
    
    # Summary by mode
    if run_hpo:
        print(f"\n{'='*80}")
        print("HPO Results:")
        print(f"{'='*80}")
        hpo_results = [r for r in all_results if r.get("mode") == "hpo"]
        for r in hpo_results:
            status = "✅" if r["success"] else "❌" if r["success"] is False else "⚠️"
            minutes = r['time_seconds'] / 60
            print(f"  {status} {r['dataset']:30s} - {minutes:.1f}m ({r['time_seconds']:.1f}s)")
    
    if run_basic:
        print(f"\n{'='*80}")
        print("Basic Training Results:")
        print(f"{'='*80}")
        basic_results = [r for r in all_results if r.get("mode") == "basic"]
        for r in basic_results:
            status = "✅" if r["success"] else "❌" if r["success"] is False else "⚠️"
            minutes = r['time_seconds'] / 60
            line = f"  {status} {r['dataset']:30s} - {minutes:.1f}m ({r['time_seconds']:.1f}s)"
            if r["success"] and r.get("result"):
                test_rmse = r["result"].get("test_metrics", {}).get("rmse", "N/A")
                if isinstance(test_rmse, float):
                    line += f" | Test RMSE: {test_rmse:.6f}"
            print(line)
    
    # Overall statistics
    successful = sum(1 for r in all_results if r["success"] is True)
    failed = sum(1 for r in all_results if r["success"] is False)
    interrupted = sum(1 for r in all_results if r["success"] is None)
    
    print(f"\n{'='*80}")
    print(f"Overall: ✅ {successful} successful | ❌ {failed} failed | ⚠️  {interrupted} interrupted")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

