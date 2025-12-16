#!/usr/bin/env python3
"""
Foundation Model HPO Runner
===========================

Run hyperparameter optimization for foundation models (Morgan-FP, ChemBERTa, BioMed, MolCLR, MolE).

Usage:
    # Run HPO on single model + dataset
    python scripts/run_foundation_hpo.py --model morgan --dataset Caco2_Wang --algo pso --trials 20
    
    # Run with config file
    python scripts/run_foundation_hpo.py --config config_foundation_benchmark.yaml
    
    # List available models and datasets
    python scripts/run_foundation_hpo.py --list
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from optimization.foundation_space import bounds, decode_vector, get_dimension
from optimization.foundation_problem import build_foundation_problem
from optimization.foundation_runner import train_foundation_model_with_best, save_foundation_results

# Import HPO algorithms from NiaPy directly
# Import HPO algorithms from NiaPy directly
from niapy.algorithms.basic import (
    ParticleSwarmAlgorithm as PSO,
    GeneticAlgorithm as GA,
    ArtificialBeeColonyAlgorithm as ABC,
)
# Some algorithms might be in 'other' module depending on version
try:
    from niapy.algorithms.basic import SimulatedAnnealing as SA
except ImportError:
    from niapy.algorithms.other import SimulatedAnnealing as SA

try:
    from niapy.algorithms.basic import HillClimbAlgorithm as HC
except ImportError:
    from niapy.algorithms.other import HillClimbAlgorithm as HC
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

AVAILABLE_MODELS = ["morgan", "chemberta", "biomed", "molclr", "mole"]
AVAILABLE_DATASETS = [
    # ADME (regression)
    "Caco2_Wang",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    # Toxicity (classification)
    "tox21",
    "herg",
    "clintox",
]


# Custom RandomSearch implementation
class RandomSearch:
    """Simple random search algorithm."""
    def __init__(self, problem, max_iters, seed=42):
        self.problem = problem
        self.max_iters = max_iters
        self.seed = seed
        
    def run(self, problem=None):
        """Run random search and return best solution."""
        rng = np.random.RandomState(self.seed)
        lb, ub = self.problem.lower, self.problem.upper
        
        best_f, best_x = float('inf'), None
        for i in range(self.max_iters):
            x = lb + rng.rand(len(lb)) * (ub - lb)
            f = self.problem._evaluate(x)
            if f < best_f:
                best_f, best_x = f, x
            if (i + 1) % 5 == 0:
                print(f"  Trial {i+1}/{self.max_iters}: val_loss={f:.4f}, best={best_f:.4f}")
        
        return best_x, best_f


AVAILABLE_ALGORITHMS = {
    "random": RandomSearch,
    "pso": PSO,
    "ga": GA,
    "sa": SA,
    "hc": HC,
    "abc": ABC,
}

DEFAULT_TRIALS = 30
DEFAULT_MAX_ITER = 500
DEFAULT_SEED = 42


# ============================================================================
# HPO EXECUTION
# ============================================================================

def run_foundation_hpo(
    model_type: str,
    dataset: str,
    algo_name: str = "random",
    trials: int = DEFAULT_TRIALS,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    output_dir: str = "runs/foundation",
    verbose: bool = True,
):
    """
    Run HPO for a foundation model on a dataset.
    
    Args:
        model_type: Type of foundation model
        dataset: Dataset name
        algo_name: HPO algorithm name
        trials: Number of trials
        max_iter: Max iterations for sklearn predictor
        seed: Random seed
        device: Device for encoder
        output_dir: Output directory for results
        verbose: Print progress
        
    Returns:
        Path to saved results file
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Foundation Model HPO: {model_type.upper()} on {dataset}")
        print(f"{'='*80}")
        print(f"Algorithm: {algo_name}")
        print(f"Trials: {trials}")
        print(f"Max iterations: {max_iter}")
        print(f"Device: {device}")
        print(f"Seed: {seed}\n")
    
    # Build problem
    if verbose:
        print("Building optimization problem...")
    
    problem = build_foundation_problem(
        model_type=model_type,
        dataset=dataset,
        max_iter=max_iter,
        seed=seed,
        device=device,
        verbose=verbose,
    )
    
    # Select algorithm
    if algo_name not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(AVAILABLE_ALGORITHMS.keys())}")
    
    AlgoClass = AVAILABLE_ALGORITHMS[algo_name]
    
    if verbose:
        print(f"\nRunning {algo_name.upper()} optimization...")
    
    # Run optimization
    if algo_name == "random":
        algorithm = AlgoClass(problem=problem, max_iters=trials, seed=seed)
    else:
        # NiaPy algorithms use different API
        algorithm = AlgoClass(seed=seed)
        algorithm.set_parameters(n_fes=trials * 10)  # Function evaluations
    
    best_x, best_fitness = algorithm.run(problem)
    
    # Decode best hyperparameters
    best_params = decode_vector(best_x, model_type)
    
    if verbose:
        print(f"\nBest validation loss: {best_fitness:.6f}")
        print(f"Best hyperparameters: {best_params}")
    
    # Train with best hyperparameters
    if verbose:
        print(f"\n{'='*80}")
        print("Training with best hyperparameters...")
        print(f"{'='*80}")
    
    result = train_foundation_model_with_best(
        model_type=model_type,
        best_params=best_params,
        dataset=dataset,
        max_iter=max_iter,
        seed=seed,
        device=device,
        verbose=verbose,
    )
    
    # Save results
    output_path = save_foundation_results(
        model_type=model_type,
        dataset=dataset,
        algo_name=algo_name,
        best_params=best_params,
        best_val_loss=best_fitness,
        trials=trials,
        result=result,
        out_dir=output_dir,
        seed=seed,
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}\n")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run foundation model HPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Morgan-FP HPO on Caco2_Wang with PSO
  python scripts/run_foundation_hpo.py --model morgan --dataset Caco2_Wang --algo pso --trials 20

  # Run ChemBERTa HPO on Half_Life_Obach with Random Search
  python scripts/run_foundation_hpo.py --model chemberta --dataset Half_Life_Obach --algo random --trials 30

  # List available options
  python scripts/run_foundation_hpo.py --list
        """
    )
    
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS,
                       help="Foundation model type")
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS,
                       help="Dataset name")
    parser.add_argument("--algo", type=str, default="random",
                       choices=list(AVAILABLE_ALGORITHMS.keys()),
                       help="HPO algorithm (default: random)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                       help=f"Number of trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                       help=f"Max iterations for predictor (default: {DEFAULT_MAX_ITER})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device (default: auto)")
    parser.add_argument("--output-dir", type=str, default="runs/foundation",
                       help="Output directory (default: runs/foundation)")
    parser.add_argument("--list", action="store_true",
                       help="List available models and datasets")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable Models:")
        for model in AVAILABLE_MODELS:
            dim = get_dimension(model)
            print(f"  - {model} (search space dim: {dim})")
        
        print("\nAvailable Datasets (ADME):")
        for ds in AVAILABLE_DATASETS[:4]:
            print(f"  - {ds}")
        
        print("\nAvailable Datasets (Toxicity):")
        for ds in AVAILABLE_DATASETS[4:]:
            print(f"  - {ds}")
        
        print("\nAvailable Algorithms:")
        for algo in AVAILABLE_ALGORITHMS.keys():
            print(f"  - {algo}")
        
        return
    
    # Validate required arguments
    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required (or use --list)")
    
    # Run HPO
    try:
        run_foundation_hpo(
            model_type=args.model,
            dataset=args.dataset,
            algo_name=args.algo,
            trials=args.trials,
            max_iter=args.max_iter,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
