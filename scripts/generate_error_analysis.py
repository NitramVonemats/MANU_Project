#!/usr/bin/env python3
"""
Error Analysis for Classification and Regression Models
========================================================
Analyzes prediction errors and correlates with molecular properties.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import torch
from optimized_gnn import (
    prepare_dataset,
    build_loaders,
    OptimizedMolecularGNN,
    OptimizedGNNConfig,
    evaluate_classification,
    evaluate,
    resolve_device,
)

FIGURES_DIR = PROJECT_ROOT / "figures" / "paper"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_molecular_properties(smiles: str):
    """Compute molecular properties for a SMILES string."""
    if not RDKIT_AVAILABLE:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    try:
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumAtoms": mol.GetNumAtoms(),
        }
    except:
        return {}


def get_predictions(dataset_name: str, config: OptimizedGNNConfig = None, seed: int = 42):
    """Get model predictions for a dataset."""
    device = resolve_device("auto")

    cfg = config or OptimizedGNNConfig(
        hidden_dim=128,
        num_layers=4,
        head_dims=(256, 128, 64),
        lr=0.001,
    )

    # Prepare data
    dataset_cache = prepare_dataset(dataset_name, val_fraction=0.1, seed=seed, verbose=False)

    loaders = build_loaders(
        dataset_name=dataset_name,
        batch_train=cfg.batch_train,
        batch_eval=cfg.batch_eval,
        val_fraction=cfg.val_fraction,
        seed=seed,
        dataset_cache=dataset_cache,
        verbose=False,
        return_cache=True,
    )
    train_loader, val_loader, test_loader, (mu, sigma), dataset_cache = loaders

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
    ).to(device)

    # Train briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    from optimized_gnn import train_epoch, is_classification_dataset
    is_cls = is_classification_dataset(dataset_name)

    for epoch in range(50):
        train_epoch(model, train_loader, optimizer, device, is_classification=is_cls)

    # Get predictions
    model.eval()
    predictions = []
    actuals = []
    smiles_list = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)

            if is_cls:
                pred_proba = torch.sigmoid(pred).cpu().numpy()
                pred_class = (pred_proba >= 0.5).astype(int)
                predictions.extend(pred_proba.flatten().tolist())
            else:
                pred_vals = pred.cpu().numpy()
                predictions.extend(pred_vals.flatten().tolist())

            actuals.extend(data.y.cpu().numpy().flatten().tolist())

    # Get SMILES from test set
    test_graphs = dataset_cache["test"]
    for g in test_graphs:
        smiles_list.append(g.smiles if hasattr(g, 'smiles') else "")

    return predictions, actuals, smiles_list, is_cls, mu, sigma


def analyze_classification_errors(dataset_name: str = "herg"):
    """Analyze false positives and negatives for classification."""
    print(f"\n{'='*50}")
    print(f"Error Analysis: {dataset_name}")
    print(f"{'='*50}")

    preds, actuals, smiles_list, is_cls, mu, sigma = get_predictions(dataset_name)

    if not is_cls:
        print(f"Skipping classification analysis for regression dataset")
        return None

    # Convert to binary predictions
    pred_proba = np.array(preds)
    pred_class = (pred_proba >= 0.5).astype(int)
    actual_class = np.array(actuals).astype(int)

    # Identify errors
    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []

    for i, (pred, actual, prob) in enumerate(zip(pred_class, actual_class, pred_proba)):
        smiles = smiles_list[i] if i < len(smiles_list) else ""
        props = compute_molecular_properties(smiles)

        entry = {
            "index": i,
            "smiles": smiles,
            "pred_proba": float(prob),
            "pred_class": int(pred),
            "actual_class": int(actual),
            **props
        }

        if pred == 1 and actual == 0:
            false_positives.append(entry)
        elif pred == 0 and actual == 1:
            false_negatives.append(entry)
        elif pred == 1 and actual == 1:
            true_positives.append(entry)
        else:
            true_negatives.append(entry)

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {len(true_positives)}")
    print(f"  True Negatives:  {len(true_negatives)}")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")

    # Analyze property distributions
    def avg_property(entries, key):
        vals = [e.get(key) for e in entries if e.get(key) is not None]
        return np.mean(vals) if vals else np.nan

    properties = ["MW", "LogP", "TPSA"]

    print(f"\nMolecular Property Analysis:")
    print(f"{'Property':<10} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10}")
    print("-" * 50)

    for prop in properties:
        tp_avg = avg_property(true_positives, prop)
        tn_avg = avg_property(true_negatives, prop)
        fp_avg = avg_property(false_positives, prop)
        fn_avg = avg_property(false_negatives, prop)
        print(f"{prop:<10} {tp_avg:<10.2f} {tn_avg:<10.2f} {fp_avg:<10.2f} {fn_avg:<10.2f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, prop in enumerate(properties):
        ax = axes[idx]

        categories = ["True Pos", "True Neg", "False Pos", "False Neg"]
        groups = [true_positives, true_negatives, false_positives, false_negatives]
        colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]

        for cat, group, color in zip(categories, groups, colors):
            vals = [e.get(prop) for e in group if e.get(prop) is not None]
            if vals:
                ax.hist(vals, bins=15, alpha=0.5, label=cat, color=color, density=True)

        ax.set_xlabel(prop, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{dataset_name} - Error Analysis by Molecular Properties", fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_path = FIGURES_DIR / f"error_analysis_{dataset_name.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {output_path}")

    return {
        "true_positives": len(true_positives),
        "true_negatives": len(true_negatives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "fp_examples": false_positives[:5],
        "fn_examples": false_negatives[:5],
    }


def analyze_regression_errors(dataset_name: str = "Caco2_Wang"):
    """Analyze high-error predictions for regression."""
    print(f"\n{'='*50}")
    print(f"Regression Error Analysis: {dataset_name}")
    print(f"{'='*50}")

    preds, actuals, smiles_list, is_cls, mu, sigma = get_predictions(dataset_name)

    if is_cls:
        print(f"Skipping regression analysis for classification dataset")
        return None

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Compute errors
    errors = np.abs(preds - actuals)
    relative_errors = errors / (np.abs(actuals) + 1e-6)

    # Get top 10% errors
    threshold = np.percentile(errors, 90)
    high_error_idx = np.where(errors >= threshold)[0]

    print(f"\nTop 10% high-error predictions:")
    print(f"  Count: {len(high_error_idx)}")
    print(f"  Error threshold: {threshold:.4f}")

    # Analyze properties of high-error molecules
    high_error_entries = []
    low_error_entries = []

    for i in range(len(preds)):
        smiles = smiles_list[i] if i < len(smiles_list) else ""
        props = compute_molecular_properties(smiles)

        entry = {
            "index": i,
            "smiles": smiles,
            "prediction": float(preds[i]),
            "actual": float(actuals[i]),
            "error": float(errors[i]),
            **props
        }

        if errors[i] >= threshold:
            high_error_entries.append(entry)
        else:
            low_error_entries.append(entry)

    # Compare properties
    def avg_property(entries, key):
        vals = [e.get(key) for e in entries if e.get(key) is not None]
        return np.mean(vals) if vals else np.nan

    properties = ["MW", "LogP", "TPSA", "NumAtoms"]

    print(f"\nMolecular Property Comparison (High vs Low Error):")
    print(f"{'Property':<12} {'High Error':<15} {'Low Error':<15}")
    print("-" * 45)

    for prop in properties:
        high_avg = avg_property(high_error_entries, prop)
        low_avg = avg_property(low_error_entries, prop)
        print(f"{prop:<12} {high_avg:<15.2f} {low_avg:<15.2f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Prediction vs Actual scatter
    ax1 = axes[0]
    ax1.scatter(actuals, preds, alpha=0.5, s=20)
    ax1.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', lw=2, label='Perfect')
    ax1.set_xlabel("Actual", fontsize=11)
    ax1.set_ylabel("Predicted", fontsize=11)
    ax1.set_title("Prediction vs Actual", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error distribution
    ax2 = axes[1]
    ax2.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(threshold, color='red', linestyle='--', lw=2, label=f'90th percentile: {threshold:.4f}')
    ax2.set_xlabel("Absolute Error", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Error Distribution", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{dataset_name} - Regression Error Analysis", fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_path = FIGURES_DIR / f"error_analysis_{dataset_name.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {output_path}")

    return {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "high_error_threshold": float(threshold),
        "high_error_count": len(high_error_entries),
    }


def main():
    print("="*60)
    print("Error Analysis for GNN Models")
    print("="*60)

    results = {}

    # Classification error analysis (hERG)
    results["herg"] = analyze_classification_errors("herg")

    # Regression error analysis (Caco2_Wang)
    results["caco2"] = analyze_regression_errors("Caco2_Wang")

    # Save results
    output_path = FIGURES_DIR / "error_analysis_summary.json"

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
    print("\n[OK] Error analysis complete!")


if __name__ == "__main__":
    main()
