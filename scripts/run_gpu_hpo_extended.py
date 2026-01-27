#!/usr/bin/env python3
"""
GPU HPO Extended Experiments
============================
Runs 50-trial HPO experiments with multiple algorithms including Optuna TPE.
Targets: hERG (classification), Caco2_Wang (regression)
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_gnn import (
    OptimizedGNNConfig,
    OptimizedMolecularGNN,
    prepare_dataset,
    build_loaders,
    evaluate,
    evaluate_classification,
    is_classification_dataset,
    resolve_device,
    train_epoch,
)

# Try to import Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] Optuna not available")

# Try to import NiaPy for other algorithms
try:
    from niapy.algorithms.basic import ParticleSwarmOptimization
    from niapy.algorithms.other import SimulatedAnnealing, HillClimbAlgorithm, RandomSearch
    from niapy.task import Task as NiaPyTask
    from niapy.problems import Problem
    NIAPY_AVAILABLE = True
except ImportError:
    NIAPY_AVAILABLE = False
    print("[WARNING] NiaPy not available")

RESULTS_DIR = PROJECT_ROOT / "results" / "hpo_extended"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter search space
SEARCH_SPACE = {
    "hidden_dim": (64, 384),
    "num_layers": (2, 8),
    "lr": (1e-4, 1e-2),
    "weight_decay": (1e-6, 1e-2),
    "head_dim_1": (64, 512),
    "head_dim_2": (32, 256),
    "head_dim_3": (16, 128),
}


def compute_class_weights(train_graphs):
    """Compute class weights for imbalanced binary classification."""
    labels = np.array([g.original_y for g in train_graphs])
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    total = len(labels)

    # Inverse frequency weighting
    weight_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
    weight_neg = total / (2 * n_neg) if n_neg > 0 else 1.0

    return weight_pos


def train_epoch_weighted(model, loader, optimizer, device, pos_weight, max_grad_norm=1.0):
    """Training epoch with class-weighted BCE for imbalanced classification."""
    import torch.nn.functional as F
    model.train()
    total_loss = 0

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    for data in loader:
        data = data.to(device)
        pred = model(data)
        targets = data.y
        loss = F.binary_cross_entropy_with_logits(pred, targets, pos_weight=pos_weight_tensor)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


class GNNTrainer:
    """Wrapper for GNN training with various HPO backends."""

    def __init__(self, dataset_name: str, device: str = "auto", seed: int = 42, verbose: bool = False, use_class_weights: bool = True):
        self.dataset_name = dataset_name
        self.device = resolve_device(device)
        self.seed = seed
        self.verbose = verbose
        self.is_classification = is_classification_dataset(dataset_name)
        self.use_class_weights = use_class_weights

        # Prepare dataset once
        self.dataset_cache = prepare_dataset(
            dataset_name=dataset_name,
            val_fraction=0.1,
            seed=seed,
            verbose=verbose,
        )

        # Compute class weights for imbalanced classification
        self.pos_weight = 1.0
        if self.is_classification and self.use_class_weights:
            self.pos_weight = compute_class_weights(self.dataset_cache["train"])
            if verbose:
                print(f"Using class weights: pos_weight={self.pos_weight:.3f}")

        # Store timing info
        self.trial_times = []

    def train_and_evaluate(self, params: Dict[str, Any], epochs: int = 100, patience: int = 20) -> Dict[str, float]:
        """Train model with given params and return validation metric."""
        start_time = time.time()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build config
        head_dims = (
            int(params.get("head_dim_1", 256)),
            int(params.get("head_dim_2", 128)),
            int(params.get("head_dim_3", 64)),
        )

        cfg = OptimizedGNNConfig(
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            head_dims=head_dims,
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            batch_train=32,
            batch_eval=64,
        )

        # Build loaders
        loaders = build_loaders(
            dataset_name=self.dataset_name,
            batch_train=cfg.batch_train,
            batch_eval=cfg.batch_eval,
            val_fraction=cfg.val_fraction,
            seed=self.seed,
            dataset_cache=self.dataset_cache,
            verbose=False,
            return_cache=True,
        )
        train_loader, val_loader, test_loader, (mu, sigma), _ = loaders

        # Build model
        sample_graph = self.dataset_cache["train"][0]
        input_dim = int(sample_graph.x.size(-1))
        adme_dim = int(sample_graph.adme_features.numel())

        model = OptimizedMolecularGNN(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            adme_dim=adme_dim,
            head_dims=cfg.head_dims,
            dropout=0.0,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        best_val_score = float("-inf") if self.is_classification else float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train - use weighted BCE for classification if enabled
            if self.is_classification and self.use_class_weights:
                train_loss = train_epoch_weighted(
                    model, train_loader, optimizer, self.device,
                    self.pos_weight, max_grad_norm=1.0
                )
            else:
                train_loss = train_epoch(
                    model, train_loader, optimizer, self.device,
                    max_grad_norm=1.0, is_classification=self.is_classification
                )

            # Evaluate
            if self.is_classification:
                val_metrics = evaluate_classification(model, val_loader, self.device)
                val_score = val_metrics["f1"]
                scheduler.step(1 - val_score)

                # Track best
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                val_metrics = evaluate(model, val_loader, self.device, mu, sigma)
                val_score = -val_metrics["rmse"]  # Negative because we maximize
                scheduler.step(-val_score)

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                break

        # Load best model and evaluate on test
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        if self.is_classification:
            val_final = evaluate_classification(model, val_loader, self.device)
            test_final = evaluate_classification(model, test_loader, self.device)
            result = {
                "val_f1": val_final["f1"],
                "val_auc": val_final["auc_roc"],
                "test_f1": test_final["f1"],
                "test_auc": test_final["auc_roc"],
            }
        else:
            val_final = evaluate(model, val_loader, self.device, mu, sigma)
            test_final = evaluate(model, test_loader, self.device, mu, sigma)
            result = {
                "val_rmse": val_final["rmse"],
                "val_r2": val_final["r2"],
                "test_rmse": test_final["rmse"],
                "test_r2": test_final["r2"],
            }

        elapsed = time.time() - start_time
        result["training_time"] = elapsed
        self.trial_times.append(elapsed)

        return result


def run_optuna_hpo(trainer: GNNTrainer, n_trials: int = 50) -> Dict[str, Any]:
    """Run HPO with Optuna TPE sampler."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed")

    trial_history = []

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 384, step=32),
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "head_dim_1": trial.suggest_int("head_dim_1", 64, 512, step=32),
            "head_dim_2": trial.suggest_int("head_dim_2", 32, 256, step=32),
            "head_dim_3": trial.suggest_int("head_dim_3", 16, 128, step=16),
        }

        result = trainer.train_and_evaluate(params)

        # Track history
        if trainer.is_classification:
            score = result["val_f1"]
        else:
            score = -result["val_rmse"]  # Maximize (negative RMSE)

        trial_history.append({
            "trial": trial.number + 1,
            "params": params,
            "val_score": score,
            "result": result,
        })

        print(f"  [Optuna] Trial {trial.number + 1}/{n_trials}: score={score:.4f}")

        return score

    # Create study
    sampler = TPESampler(seed=trainer.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    direction = "maximize"  # We converted RMSE to negative
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trial_history": trial_history,
        "n_trials": n_trials,
        "importance": optuna.importance.get_param_importances(study) if len(study.trials) > 5 else {},
    }


def run_niapy_hpo(trainer: GNNTrainer, algorithm: str, n_trials: int = 50) -> Dict[str, Any]:
    """Run HPO with NiaPy algorithms (PSO, SA, Random)."""
    if not NIAPY_AVAILABLE:
        raise ImportError("NiaPy not installed")

    trial_history = []
    trial_counter = [0]
    best_so_far = [float("-inf")]

    class GNNProblem(Problem):
        def __init__(self, trainer_ref):
            self.trainer = trainer_ref
            # 7 dimensions: hidden_dim, num_layers, lr (log), wd (log), head1, head2, head3
            super().__init__(dimension=7, lower=0.0, upper=1.0)

        def _evaluate(self, x):
            # Decode normalized [0,1] to actual values
            params = {
                "hidden_dim": int(64 + x[0] * (384 - 64)),
                "num_layers": int(2 + x[1] * (8 - 2)),
                "lr": 10 ** (-4 + x[2] * (-2 - (-4))),  # log scale
                "weight_decay": 10 ** (-6 + x[3] * (-2 - (-6))),  # log scale
                "head_dim_1": int(64 + x[4] * (512 - 64)),
                "head_dim_2": int(32 + x[5] * (256 - 32)),
                "head_dim_3": int(16 + x[6] * (128 - 16)),
            }

            result = self.trainer.train_and_evaluate(params)

            if self.trainer.is_classification:
                score = result["val_f1"]
                fitness = -score  # Minimize negative F1
            else:
                score = -result["val_rmse"]
                fitness = result["val_rmse"]  # Minimize RMSE

            trial_counter[0] += 1
            best_so_far[0] = max(best_so_far[0], score)

            trial_history.append({
                "trial": trial_counter[0],
                "params": params,
                "val_score": score,
                "best_so_far": best_so_far[0],
                "result": result,
            })

            print(f"  [{algorithm.upper()}] Trial {trial_counter[0]}/{n_trials}: score={score:.4f}, best={best_so_far[0]:.4f}")

            return fitness

    problem = GNNProblem(trainer)

    # Select algorithm
    if algorithm == "pso":
        algo = ParticleSwarmOptimization(population_size=10, seed=trainer.seed)
    elif algorithm == "sa":
        algo = SimulatedAnnealing(seed=trainer.seed)
    elif algorithm == "random":
        algo = RandomSearch(seed=trainer.seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Run optimization (limit to n_trials evaluations)
    task = NiaPyTask(problem=problem, max_evals=n_trials)
    best_x, best_fitness = algo.run(task)

    # Decode best params
    best_params = {
        "hidden_dim": int(64 + best_x[0] * (384 - 64)),
        "num_layers": int(2 + best_x[1] * (8 - 2)),
        "lr": 10 ** (-4 + best_x[2] * (-2 - (-4))),
        "weight_decay": 10 ** (-6 + best_x[3] * (-2 - (-6))),
        "head_dim_1": int(64 + best_x[4] * (512 - 64)),
        "head_dim_2": int(32 + best_x[5] * (256 - 32)),
        "head_dim_3": int(16 + best_x[6] * (128 - 16)),
    }

    return {
        "best_params": best_params,
        "best_fitness": float(best_fitness),
        "trial_history": trial_history,
        "n_trials": len(trial_history),
    }


def run_multiseed_validation(trainer: GNNTrainer, best_params: Dict, seeds: List[int] = [42, 43, 44]):
    """Run best configuration with multiple seeds for variance estimation."""
    results = []

    for seed in seeds:
        trainer.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = trainer.train_and_evaluate(best_params, epochs=100, patience=20)
        results.append(result)

        if trainer.is_classification:
            print(f"  Seed {seed}: F1={result['test_f1']:.4f}, AUC={result['test_auc']:.4f}")
        else:
            print(f"  Seed {seed}: RMSE={result['test_rmse']:.4f}, R2={result['test_r2']:.4f}")

    # Compute statistics
    if trainer.is_classification:
        f1_scores = [r["test_f1"] for r in results]
        auc_scores = [r["test_auc"] for r in results]
        summary = {
            "test_f1_mean": np.mean(f1_scores),
            "test_f1_std": np.std(f1_scores),
            "test_auc_mean": np.mean(auc_scores),
            "test_auc_std": np.std(auc_scores),
        }
    else:
        rmse_scores = [r["test_rmse"] for r in results]
        r2_scores = [r["test_r2"] for r in results]
        summary = {
            "test_rmse_mean": np.mean(rmse_scores),
            "test_rmse_std": np.std(rmse_scores),
            "test_r2_mean": np.mean(r2_scores),
            "test_r2_std": np.std(r2_scores),
        }

    summary["raw_results"] = results
    return summary


def run_full_experiment(dataset_name: str, n_trials: int = 50, device: str = "auto"):
    """Run full HPO experiment for a dataset."""
    print(f"\n{'='*60}")
    print(f"GPU HPO Extended: {dataset_name}")
    print(f"Trials: {n_trials}, Device: {device}")
    print(f"{'='*60}")

    results = {}

    # Algorithms to test
    algorithms = ["optuna", "pso", "sa", "random"]

    for algo in algorithms:
        print(f"\n--- Algorithm: {algo.upper()} ---")

        trainer = GNNTrainer(dataset_name, device=device, seed=42, verbose=False)
        start_time = time.time()

        try:
            if algo == "optuna":
                hpo_result = run_optuna_hpo(trainer, n_trials=n_trials)
            else:
                hpo_result = run_niapy_hpo(trainer, algo, n_trials=n_trials)

            elapsed = time.time() - start_time

            # Multi-seed validation
            print(f"\n  Running multi-seed validation...")
            multiseed = run_multiseed_validation(trainer, hpo_result["best_params"])

            results[algo] = {
                "best_params": hpo_result["best_params"],
                "hpo_time": elapsed,
                "avg_trial_time": np.mean(trainer.trial_times) if trainer.trial_times else 0,
                "multiseed_summary": multiseed,
                "trial_history": hpo_result.get("trial_history", []),
            }

            if algo == "optuna" and "importance" in hpo_result:
                results[algo]["param_importance"] = hpo_result["importance"]

            print(f"\n  [OK] {algo.upper()} complete in {elapsed:.1f}s")

        except Exception as e:
            print(f"  [ERROR] {algo}: {str(e)}")
            results[algo] = {"error": str(e)}

    # Save results
    output_dir = RESULTS_DIR / dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"hpo_extended_{dataset_name.lower()}_results.json"

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GPU HPO Extended Experiments")
    parser.add_argument("--dataset", type=str, default="herg",
                       choices=["herg", "Caco2_Wang", "Half_Life_Obach",
                               "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ", "tox21"])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    results = run_full_experiment(args.dataset, n_trials=args.trials, device=args.device)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    is_cls = is_classification_dataset(args.dataset)

    for algo, data in results.items():
        if "error" in data:
            print(f"\n{algo.upper()}: ERROR - {data['error']}")
            continue

        summary = data["multiseed_summary"]

        if is_cls:
            print(f"\n{algo.upper()}:")
            print(f"  Test F1:  {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
            print(f"  Test AUC: {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
        else:
            print(f"\n{algo.upper()}:")
            print(f"  Test RMSE: {summary['test_rmse_mean']:.4f} ± {summary['test_rmse_std']:.4f}")
            print(f"  Test R²:   {summary['test_r2_mean']:.4f} ± {summary['test_r2_std']:.4f}")

        print(f"  HPO Time: {data['hpo_time']:.1f}s")
        print(f"  Avg Trial: {data['avg_trial_time']:.1f}s")


if __name__ == "__main__":
    main()
