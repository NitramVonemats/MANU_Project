"""
MODEL COMPARISON - Phase 1
==========================
Test 5 GNN architectures (GCN, GAT, SAGE, GIN, GINE) on 7 datasets
to identify the best model per dataset.

Models:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network
- GraphSAGE: Sampling and Aggregation
- GIN: Graph Isomorphism Network
- GINE: GIN with Edge features

Datasets:
- Caco2_Wang (regression)
- Half_Life_Obach (regression)
- Clearance_Hepatocyte_AZ (regression)
- Clearance_Microsome_AZ (regression)
- tox21 (classification)
- herg (classification)
- clintox (classification)

Total experiments: 5 models x 7 datasets = 35 runs
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
import pandas as pd
import numpy as np

# Import from existing optimized_gnn.py
import sys
sys.path.insert(0, str(Path(__file__).parent))

from optimized_gnn import (
    prepare_dataset,
    is_classification_dataset,
    OptimizedGNNConfig,
    resolve_device,
    SimpleReadout,
)

# Import GNN backbone implementations
from adme_gnn.models.gnn import GNNBackbone


class FlexibleGNN(nn.Module):
    """
    Flexible GNN that works with multiple architectures

    Args:
        model_type: One of ["GCN", "GAT", "SAGE", "GIN", "GINE"]
        input_dim: Input node feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        adme_dim: ADME feature dimension
        head_dims: Prediction head dimensions
        dropout: Dropout rate
        edge_dim: Edge feature dimension (for GINE)
        heads: Number of attention heads (for GAT)
    """

    def __init__(
        self,
        model_type: str,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 5,
        adme_dim: int = 15,
        head_dims: tuple = (256, 128, 64),
        dropout: float = 0.0,
        edge_dim: int = 12,
        heads: int = 4,
    ):
        super().__init__()

        self.model_type = model_type

        # GNN backbone
        self.backbone = GNNBackbone(
            model_type=model_type,
            layers=num_layers,
            hidden=hidden_dim,
            dropout=dropout,
            input_dim=input_dim,
            edge_dim=edge_dim,
            heads=heads,
        )

        # Readout (pooling)
        self.readout = SimpleReadout(hidden_dim)

        # Prediction head
        combined_dim = self.readout.out_dim + adme_dim
        self.head = self._build_head(combined_dim, head_dims, dropout)

    @staticmethod
    def _build_head(input_dim: int, head_dims: tuple, dropout: float = 0.0) -> nn.Sequential:
        """Build prediction head"""
        if not head_dims:
            raise ValueError("head_dims must contain at least one layer size")

        layers = []
        current_dim = input_dim
        for hidden in head_dims:
            layers.append(nn.Linear(current_dim, int(hidden)))
            layers.append(nn.BatchNorm1d(int(hidden)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = int(hidden)
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, data):
        """Forward pass"""
        # GNN backbone
        graph_emb = self.backbone(data)

        # Pooling
        graph_pooled = self.readout(graph_emb, data.batch)

        # ADME features
        adme = data.adme_features
        if adme.dim() == 1:
            adme = adme.unsqueeze(0)

        # Combine and predict
        if adme.numel() > 0:
            combined = torch.cat([graph_pooled, adme], dim=-1)
        else:
            combined = graph_pooled

        return self.head(combined).squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, data.y)

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_regression(model, loader, device, mu, sigma, is_log_transformed=False):
    """Evaluate regression model"""
    model.eval()
    preds, trues = [], []

    if len(loader) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    for data in loader:
        data = data.to(device)
        pred = model(data)
        preds.append(pred.cpu())
        trues.append(data.y.cpu())

    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    # Check for invalid values
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    if np.any(np.isnan(trues)) or np.any(np.isinf(trues)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}

    # Inverse transform
    preds_log = preds * sigma + mu
    trues_log = trues * sigma + mu

    preds_orig = np.exp(preds_log)
    trues_orig = np.exp(trues_log)

    # Check for invalid values after inverse transform
    if np.any(np.isnan(preds_orig)) or np.any(np.isinf(preds_orig)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    if np.any(np.isnan(trues_orig)) or np.any(np.isinf(trues_orig)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}

    # Metrics
    squared_errors = (preds_orig - trues_orig) ** 2
    mse = np.mean(squared_errors)

    if mse < 1e-10:
        rmse = 0.0
    else:
        rmse = np.sqrt(mse)

    mae = np.mean(np.abs(preds_orig - trues_orig))

    # R2
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((trues_orig - trues_orig.mean()) ** 2)

    if ss_tot < 1e-12:
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        r2 = max(-10.0, min(1.0, r2))

    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


@torch.no_grad()
def evaluate_classification(model, loader, device):
    """Evaluate classification model"""
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    model.eval()
    preds, probs, trues = [], [], []

    if len(loader) == 0:
        return {
            "auc_roc": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }

    for data in loader:
        data = data.to(device)
        logit = model(data)
        prob = torch.sigmoid(logit)

        probs.append(prob.cpu())
        preds.append((prob > 0.5).float().cpu())
        trues.append(data.y.cpu())

    if not preds:
        return {
            "auc_roc": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }

    probs = torch.cat(probs).numpy()
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    # Check for invalid values
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        return {
            "auc_roc": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }

    # Calculate metrics
    try:
        auc_roc = roc_auc_score(trues, probs)
    except:
        auc_roc = float("nan")

    try:
        f1 = f1_score(trues, preds, zero_division=0)
    except:
        f1 = float("nan")

    try:
        accuracy = accuracy_score(trues, preds)
    except:
        accuracy = float("nan")

    return {
        "auc_roc": float(auc_roc),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def train_and_evaluate(
    model_type: str,
    dataset_name: str,
    dataset_cache: Dict[str, Any],
    config: OptimizedGNNConfig,
    device: str,
    epochs: int = 100,
    patience: int = 20,
    verbose: bool = True,
):
    """Train and evaluate a single model on a single dataset"""

    start_time = time.time()

    # Extract data from dataset_cache
    train_graphs = dataset_cache["train"]
    val_graphs = dataset_cache["val"]
    test_graphs = dataset_cache["test"]
    mu, sigma = dataset_cache["log_stats"]
    is_log_transformed = dataset_cache.get("is_log_transformed", False)

    # Create dataloaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=config.batch_train, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_eval, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_eval, shuffle=False)

    # Check if classification
    is_classification = is_classification_dataset(dataset_name)

    # Get input dimensions
    sample_data = next(iter(train_loader))
    input_dim = sample_data.x.size(1)
    adme_dim = sample_data.adme_features.size(-1) if sample_data.adme_features.numel() > 0 else 0

    # Create model
    model = FlexibleGNN(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        adme_dim=adme_dim,
        head_dims=config.head_dims,
        dropout=0.0,  # No dropout for baseline
        edge_dim=12,
        heads=4,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\n[MODEL] {model_type} ({config.num_layers} layers, {config.hidden_dim} hidden)")
        print(f"Head dims: {config.head_dims}, LR={config.lr:.4f}, WD={config.weight_decay:.2e}")
        print(f"Parameters: {n_params:,}")

    # Loss function
    if is_classification:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    # Training loop
    best_val_metric = float("inf") if not is_classification else float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if verbose:
        print(f"\n[TRAINING] Starting...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.max_grad_norm
        )

        # Evaluate
        if is_classification:
            val_metrics = evaluate_classification(model, val_loader, device)
            val_metric = val_metrics["auc_roc"]
        else:
            val_metrics = evaluate_regression(
                model, val_loader, device, mu, sigma, is_log_transformed
            )
            val_metric = val_metrics["rmse"]

        # Scheduler step
        scheduler.step(val_metric if not is_classification else -val_metric)

        # Check improvement
        is_better = (
            val_metric < best_val_metric if not is_classification
            else val_metric > best_val_metric
        )

        if is_better:
            best_val_metric = val_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        # Logging
        if verbose and (epoch == 1 or epoch % 10 == 0):
            if is_classification:
                print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val_auc={val_metric:.4f}")
            else:
                print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val_rmse={val_metric:.6f}, val_r2={val_metrics['r2']:.6f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    model = model.to(device)

    # Final evaluation
    if is_classification:
        val_metrics = evaluate_classification(model, val_loader, device)
        test_metrics = evaluate_classification(model, test_loader, device)
    else:
        val_metrics = evaluate_regression(
            model, val_loader, device, mu, sigma, is_log_transformed
        )
        test_metrics = evaluate_regression(
            model, test_loader, device, mu, sigma, is_log_transformed
        )

    train_time = time.time() - start_time

    if verbose:
        print(f"\n[OK] Training complete ({train_time:.1f}s)")
        if is_classification:
            print(f"Val  - AUC-ROC: {val_metrics['auc_roc']:.6f}, F1: {val_metrics['f1']:.6f}, Acc: {val_metrics['accuracy']:.6f}")
            print(f"Test - AUC-ROC: {test_metrics['auc_roc']:.6f}, F1: {test_metrics['f1']:.6f}, Acc: {test_metrics['accuracy']:.6f}")
        else:
            print(f"Val  - RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, R2: {val_metrics['r2']:.6f}")
            print(f"Test - RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}, R2: {test_metrics['r2']:.6f}")

    # Return results
    return {
        "model_type": model_type,
        "dataset": dataset_name,
        "n_params": n_params,
        "train_time": train_time,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "is_classification": is_classification,
    }


def run_model_comparison(
    models: List[str] = None,
    datasets: List[str] = None,
    epochs: int = 100,
    patience: int = 20,
    device: str = "auto",
    seed: int = 42,
    save_dir: str = "reports/model_comparison",
    # Hyperparameter grid
    num_layers_list: List[int] = None,
    hidden_dims_list: List[int] = None,
    learning_rates: List[float] = None,
    dropouts: List[float] = None,
    head_dims_list: List[tuple] = None,
):
    """
    Run comprehensive model comparison with hyperparameter search

    Args:
        models: List of model types to test (default: all 5)
        datasets: List of datasets to test (default: all 7)
        epochs: Maximum epochs per run
        patience: Early stopping patience
        device: Device to use
        seed: Random seed
        save_dir: Directory to save results
        num_layers_list: List of layer counts to test (e.g., [3, 5, 7])
        hidden_dims_list: List of hidden dimensions (e.g., [64, 128, 256])
        learning_rates: List of learning rates (e.g., [1e-3, 1e-4])
        dropouts: List of dropout rates (e.g., [0.0, 0.3, 0.5])
        head_dims_list: List of head dimension tuples (e.g., [(256, 128, 64), (128, 64)])
    """

    # Default models and datasets
    if models is None:
        models = ["GCN", "GAT", "SAGE", "GIN", "GINE"]
    if datasets is None:
        datasets = [
            # ADME (regression)
            "Caco2_Wang",
            "Half_Life_Obach",
            "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ",
            # Tox (classification)
            "tox21",
            "herg",
            "clintox",
        ]

    # Default hyperparameter grid
    if num_layers_list is None:
        num_layers_list = [3, 5, 7]
    if hidden_dims_list is None:
        hidden_dims_list = [64, 128, 256]
    if learning_rates is None:
        learning_rates = [1e-3, 1e-4]
    if dropouts is None:
        dropouts = [0.0, 0.3, 0.5]
    if head_dims_list is None:
        head_dims_list = [(256, 128, 64), (128, 64), (64, 32)]

    device = resolve_device(device)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    save_path = Path(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_path / timestamp
    save_path.mkdir(parents=True, exist_ok=True)

    # Calculate total experiments
    total_experiments = (
        len(models) * len(datasets) * len(num_layers_list) *
        len(hidden_dims_list) * len(learning_rates) *
        len(dropouts) * len(head_dims_list)
    )

    print("=" * 70)
    print(f"MODEL COMPARISON - Phase 1 (COMPREHENSIVE)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {', '.join(models)} ({len(models)} total)")
    print(f"Datasets: {', '.join(datasets)} ({len(datasets)} total)")
    print(f"Hyperparameters:")
    print(f"  - Layers: {num_layers_list} ({len(num_layers_list)} values)")
    print(f"  - Hidden dims: {hidden_dims_list} ({len(hidden_dims_list)} values)")
    print(f"  - Learning rates: {learning_rates} ({len(learning_rates)} values)")
    print(f"  - Dropouts: {dropouts} ({len(dropouts)} values)")
    print(f"  - Head dims: {len(head_dims_list)} configurations")
    print(f"Total experiments: {total_experiments}")
    print(f"Results will be saved to: {save_path}")
    print("=" * 70)

    # Results storage
    all_results = []
    experiment_count = 0

    # Main loop: iterate through all combinations
    for dataset_idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'=' * 70}")
        print(f"Dataset {dataset_idx}/{len(datasets)}: {dataset_name}")
        print(f"{'=' * 70}")

        # Prepare dataset once (reuse for all hyperparameter combinations)
        try:
            dataset_cache = prepare_dataset(
                dataset_name=dataset_name,
                val_fraction=0.1,  # Fixed validation fraction
                seed=seed,
                verbose=True,
            )
        except Exception as e:
            print(f"[ERROR] Failed to prepare dataset {dataset_name}: {e}")
            continue

        # Test all hyperparameter combinations
        for model_type in models:
            for num_layers in num_layers_list:
                for hidden_dim in hidden_dims_list:
                    for lr in learning_rates:
                        for dropout in dropouts:
                            for head_dims in head_dims_list:
                                experiment_count += 1

                                print(f"\n{'-' * 70}")
                                print(f"Experiment {experiment_count}/{total_experiments}")
                                print(f"Dataset: {dataset_name} | Model: {model_type}")
                                print(f"Layers: {num_layers} | Hidden: {hidden_dim} | LR: {lr} | Dropout: {dropout}")
                                print(f"Head dims: {head_dims}")
                                print(f"{'-' * 70}")

                                # Create config for this run
                                config = OptimizedGNNConfig(
                                    hidden_dim=hidden_dim,
                                    num_layers=num_layers,
                                    head_dims=head_dims,
                                    lr=lr,
                                    weight_decay=0.0,
                                    batch_train=32,
                                    batch_eval=64,
                                )

                                try:
                                    result = train_and_evaluate(
                                        model_type=model_type,
                                        dataset_name=dataset_name,
                                        dataset_cache=dataset_cache,
                                        config=config,
                                        device=device,
                                        epochs=epochs,
                                        patience=patience,
                                        verbose=False,  # Less verbose for many runs
                                    )

                                    # Add hyperparameters to result
                                    result["hyperparameters"] = {
                                        "num_layers": num_layers,
                                        "hidden_dim": hidden_dim,
                                        "learning_rate": lr,
                                        "dropout": dropout,
                                        "head_dims": head_dims,
                                    }

                                    all_results.append(result)

                                    # Print summary
                                    if result["is_classification"]:
                                        print(f"[OK] Test AUC-ROC: {result['test_metrics']['auc_roc']:.4f} | "
                                              f"Val AUC-ROC: {result['val_metrics']['auc_roc']:.4f} | "
                                              f"Time: {result['train_time']:.1f}s")
                                    else:
                                        print(f"[OK] Test RMSE: {result['test_metrics']['rmse']:.4f} | "
                                              f"Test R2: {result['test_metrics']['r2']:.4f} | "
                                              f"Time: {result['train_time']:.1f}s")

                                    # Save intermediate results after each experiment
                                    results_file = save_path / "results.json"
                                    with open(results_file, "w") as f:
                                        json.dump(all_results, f, indent=2)

                                except Exception as e:
                                    print(f"[ERROR] Failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    continue

    # Generate summary report
    print(f"\n{'=' * 70}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'=' * 70}")

    # Check if we have any results
    if not all_results:
        print("[WARNING] No successful experiments! All runs failed.")
        return {
            "all_results": [],
            "summary": pd.DataFrame(),
            "best_models": pd.DataFrame(),
            "save_path": str(save_path),
        }

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Create summary by dataset
    summary_data = []
    for dataset_name in datasets:
        dataset_results = df[df["dataset"] == dataset_name]
        if len(dataset_results) == 0:
            continue

        is_classification = dataset_results.iloc[0]["is_classification"]

        for _, row in dataset_results.iterrows():
            model_type = row["model_type"]
            hparams = row.get("hyperparameters", {})

            base_info = {
                "Dataset": dataset_name,
                "Model": model_type,
                "Task": "Classification" if is_classification else "Regression",
                "Layers": hparams.get("num_layers", "N/A"),
                "Hidden_Dim": hparams.get("hidden_dim", "N/A"),
                "Learning_Rate": hparams.get("learning_rate", "N/A"),
                "Dropout": hparams.get("dropout", "N/A"),
                "Head_Dims": str(hparams.get("head_dims", "N/A")),
                "Train_Time_s": row["train_time"],
                "Best_Epoch": row["best_epoch"],
                "N_Params": row["n_params"],
            }

            if is_classification:
                test_auc = row["test_metrics"]["auc_roc"]
                test_f1 = row["test_metrics"]["f1"]
                val_auc = row["val_metrics"]["auc_roc"]

                summary_data.append({
                    **base_info,
                    "Test_AUC": test_auc,
                    "Test_F1": test_f1,
                    "Val_AUC": val_auc,
                })
            else:
                test_rmse = row["test_metrics"]["rmse"]
                test_r2 = row["test_metrics"]["r2"]
                val_rmse = row["val_metrics"]["rmse"]

                summary_data.append({
                    **base_info,
                    "Test_RMSE": test_rmse,
                    "Test_R2": test_r2,
                    "Val_RMSE": val_rmse,
                })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = save_path / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

    # Identify best model + hyperparameter combination per dataset
    best_models = []
    for dataset_name in datasets:
        dataset_summary = summary_df[summary_df["Dataset"] == dataset_name]
        if len(dataset_summary) == 0:
            continue

        task = dataset_summary.iloc[0]["Task"]

        if task == "Classification":
            # Best = highest test AUC-ROC
            best_row = dataset_summary.loc[dataset_summary["Test_AUC"].idxmax()]
            best_models.append({
                "Dataset": dataset_name,
                "Task": task,
                "Best_Model": best_row["Model"],
                "Layers": best_row["Layers"],
                "Hidden_Dim": best_row["Hidden_Dim"],
                "Learning_Rate": best_row["Learning_Rate"],
                "Dropout": best_row["Dropout"],
                "Head_Dims": best_row["Head_Dims"],
                "Test_AUC": best_row["Test_AUC"],
                "Test_F1": best_row["Test_F1"],
            })
        else:
            # Best = lowest test RMSE
            best_row = dataset_summary.loc[dataset_summary["Test_RMSE"].idxmin()]
            best_models.append({
                "Dataset": dataset_name,
                "Task": task,
                "Best_Model": best_row["Model"],
                "Layers": best_row["Layers"],
                "Hidden_Dim": best_row["Hidden_Dim"],
                "Learning_Rate": best_row["Learning_Rate"],
                "Dropout": best_row["Dropout"],
                "Head_Dims": best_row["Head_Dims"],
                "Test_RMSE": best_row["Test_RMSE"],
                "Test_R2": best_row["Test_R2"],
            })

    best_models_df = pd.DataFrame(best_models)
    best_models_file = save_path / "best_models_per_dataset.csv"
    best_models_df.to_csv(best_models_file, index=False)

    # Print summary
    print(f"\n{'=' * 70}")
    print("BEST MODEL PER DATASET")
    print(f"{'=' * 70}")
    print(best_models_df.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to: {save_path}")
    print(f"- results.json: Full results")
    print(f"- summary.csv: Summary table")
    print(f"- best_models_per_dataset.csv: Best model per dataset")

    return {
        "all_results": all_results,
        "summary": summary_df,
        "best_models": best_models_df,
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    # Run full model comparison
    results = run_model_comparison(
        epochs=100,
        patience=20,
        device="auto",
        seed=42,
    )
