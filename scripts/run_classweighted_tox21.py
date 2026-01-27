#!/usr/bin/env python3
"""
Class-Weighted BCE Experiment for Tox21
========================================
Tests the impact of class weighting on the severely imbalanced Tox21 dataset.
Tox21 (NR-AR) has ~3.8% positive samples.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_gnn import (
    OptimizedGNNConfig,
    OptimizedMolecularGNN,
    prepare_dataset,
    build_loaders,
    evaluate_classification,
    is_classification_dataset,
    resolve_device,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "class_weighted"
FIGURES_DIR = PROJECT_ROOT / "figures" / "paper"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_class_weights(train_graphs):
    """Compute class weights for imbalanced binary classification."""
    labels = np.array([g.original_y for g in train_graphs])
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    total = len(labels)

    # Inverse frequency weighting
    weight_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
    weight_neg = total / (2 * n_neg) if n_neg > 0 else 1.0

    print(f"Class distribution: Positive={n_pos} ({100*n_pos/total:.2f}%), Negative={n_neg} ({100*n_neg/total:.2f}%)")
    print(f"Class weights: pos_weight={weight_pos:.3f}, neg_weight={weight_neg:.3f}")

    return weight_pos, weight_neg


def train_epoch_weighted(model, loader, optimizer, device, pos_weight, max_grad_norm=1.0):
    """Training epoch with class-weighted BCE."""
    model.train()
    total_loss = 0

    # Create pos_weight tensor for BCEWithLogitsLoss
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    for data in loader:
        data = data.to(device)
        pred = model(data)
        targets = data.y

        # Weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, targets, pos_weight=pos_weight_tensor)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch_unweighted(model, loader, optimizer, device, max_grad_norm=1.0):
    """Training epoch with standard BCE (no weighting)."""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        targets = data.y
        loss = F.binary_cross_entropy_with_logits(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def run_experiment(
    dataset_name: str = "tox21",
    use_class_weights: bool = True,
    epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
    config: OptimizedGNNConfig = None,
):
    """Run a single training experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device("auto")
    cfg = config or OptimizedGNNConfig(
        hidden_dim=128,
        num_layers=4,
        head_dims=(256, 128, 96),
        lr=0.0003,
        weight_decay=1.14e-04,
        batch_train=32,
        batch_eval=64,
    )

    # Prepare dataset
    dataset_cache = prepare_dataset(
        dataset_name=dataset_name,
        val_fraction=cfg.val_fraction,
        seed=seed,
        verbose=True,
    )

    loaders = build_loaders(
        dataset_name=dataset_name,
        batch_train=cfg.batch_train,
        batch_eval=cfg.batch_eval,
        val_fraction=cfg.val_fraction,
        seed=seed,
        dataset_cache=dataset_cache,
        verbose=True,
        return_cache=True,
    )
    train_loader, val_loader, test_loader, (mu, sigma), dataset_cache = loaders

    # Compute class weights
    if use_class_weights:
        pos_weight, neg_weight = compute_class_weights(dataset_cache["train"])
    else:
        pos_weight = 1.0

    # Build model
    sample_graph = dataset_cache["train"][0]
    input_dim = int(sample_graph.x.size(-1))
    adme_dim = int(sample_graph.adme_features.numel())

    model = OptimizedMolecularGNN(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        adme_dim=adme_dim,
        head_dims=cfg.head_dims,
        dropout=0.0,
    ).to(device)

    print(f"\n[MODEL] Graph ({cfg.num_layers} layers, {cfg.hidden_dim} hidden)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Class-weighted BCE: {use_class_weights} (pos_weight={pos_weight:.3f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # Training
    best_val_f1 = -1
    best_model_state = None
    patience_counter = 0
    history = []

    print("\n[TRAINING] Starting...")

    for epoch in range(1, epochs + 1):
        # Train
        if use_class_weights:
            train_loss = train_epoch_weighted(model, train_loader, optimizer, device, pos_weight)
        else:
            train_loss = train_epoch_unweighted(model, train_loader, optimizer, device)

        # Evaluate
        val_metrics = evaluate_classification(model, val_loader, device)
        scheduler.step(1 - val_metrics["f1"])  # Minimize (1 - F1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc_roc"],
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val_F1={val_metrics['f1']:.4f}, val_AUC={val_metrics['auc_roc']:.4f}")

        # Early stopping
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    val_final = evaluate_classification(model, val_loader, device)
    test_final = evaluate_classification(model, test_loader, device)

    print(f"\n[FINAL RESULTS]")
    print(f"Val  - F1: {val_final['f1']:.4f}, AUC: {val_final['auc_roc']:.4f}, Precision: {val_final['precision']:.4f}, Recall: {val_final['recall']:.4f}")
    print(f"Test - F1: {test_final['f1']:.4f}, AUC: {test_final['auc_roc']:.4f}, Precision: {test_final['precision']:.4f}, Recall: {test_final['recall']:.4f}")

    return {
        "use_class_weights": use_class_weights,
        "pos_weight": pos_weight,
        "best_val_f1": best_val_f1,
        "val_metrics": val_final,
        "test_metrics": test_final,
        "history": history,
    }


def main():
    print("=" * 60)
    print("Class-Weighted BCE Experiment for Tox21")
    print("=" * 60)

    results = {}

    # Run without class weighting (baseline)
    print("\n" + "=" * 40)
    print("BASELINE: Unweighted BCE")
    print("=" * 40)
    results["unweighted"] = run_experiment(use_class_weights=False, seed=42)

    # Run with class weighting
    print("\n" + "=" * 40)
    print("EXPERIMENT: Class-Weighted BCE")
    print("=" * 40)
    results["weighted"] = run_experiment(use_class_weights=True, seed=42)

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    uw = results["unweighted"]
    w = results["weighted"]

    print(f"\n{'Metric':<20} {'Unweighted':<15} {'Weighted':<15} {'Difference':<15}")
    print("-" * 65)

    for metric in ["f1", "auc_roc", "precision", "recall"]:
        uw_val = uw["test_metrics"][metric]
        w_val = w["test_metrics"][metric]
        diff = w_val - uw_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{metric:<20} {uw_val:<15.4f} {w_val:<15.4f} {diff_str:<15}")

    # Save results
    output_path = RESULTS_DIR / "classweighted_tox21_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "unweighted": {
                "val_f1": uw["best_val_f1"],
                "test_f1": uw["test_metrics"]["f1"],
                "test_auc": uw["test_metrics"]["auc_roc"],
                "test_precision": uw["test_metrics"]["precision"],
                "test_recall": uw["test_metrics"]["recall"],
            },
            "weighted": {
                "pos_weight": w["pos_weight"],
                "val_f1": w["best_val_f1"],
                "test_f1": w["test_metrics"]["f1"],
                "test_auc": w["test_metrics"]["auc_roc"],
                "test_precision": w["test_metrics"]["precision"],
                "test_recall": w["test_metrics"]["recall"],
            }
        }, f, indent=2)

    print(f"\nResults saved: {output_path}")

    # Generate LaTeX snippet for paper
    latex_snippet = f"""
% Class-weighted BCE results for Tox21
% Unweighted: F1={uw['test_metrics']['f1']:.3f}, AUC={uw['test_metrics']['auc_roc']:.3f}
% Weighted (pos_weight={w['pos_weight']:.2f}): F1={w['test_metrics']['f1']:.3f}, AUC={w['test_metrics']['auc_roc']:.3f}
% Improvement: F1 {'+' if w['test_metrics']['f1'] > uw['test_metrics']['f1'] else ''}{w['test_metrics']['f1'] - uw['test_metrics']['f1']:.3f}
"""
    latex_path = FIGURES_DIR / "classweighted_results.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_snippet)

    print(f"LaTeX snippet saved: {latex_path}")
    print("\n[OK] Class-weighted experiment complete!")


if __name__ == "__main__":
    main()
