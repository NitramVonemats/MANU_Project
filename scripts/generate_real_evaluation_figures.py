"""
Generate REAL evaluation figures from actual model predictions.
NO synthetic data - all plots from held-out test set predictions.

Phase 2 of paper preparation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, roc_auc_score
)

from optimized_gnn import (
    prepare_dataset, build_loaders, OptimizedMolecularGNN,
    OptimizedGNNConfig, is_classification_dataset, train_model
)


def get_predictions(model, loader, device, is_classification=True):
    """Get predictions and true labels from a model."""
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)

            if is_classification:
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
            else:
                all_preds.append(logits.cpu().numpy())

            all_trues.append(data.y.cpu().numpy())

    all_trues = np.concatenate(all_trues)
    all_preds = np.concatenate(all_preds)

    if is_classification:
        all_probs = np.concatenate(all_probs)
        return all_trues, all_preds, all_probs
    else:
        return all_trues, all_preds, None


def train_and_get_predictions(dataset_name, seed=42, device='cpu'):
    """Train model and return test predictions."""
    print(f"\n{'='*70}")
    print(f"Training model for {dataset_name}")
    print(f"{'='*70}")

    is_class = is_classification_dataset(dataset_name)

    # Use default config (or load from HPO results if available)
    config = OptimizedGNNConfig(
        hidden_dim=128,
        num_layers=5,
        lr=0.001,
        batch_train=32,
        batch_eval=64,
    )

    # Train model
    result = train_model(
        dataset_name=dataset_name,
        config=config,
        epochs=100,
        patience=20,
        device=device,
        seed=seed,
        return_model=True,
        verbose=True
    )

    model = result['model']

    # Get test predictions
    cache = prepare_dataset(dataset_name, val_fraction=0.1, seed=seed, verbose=False)
    _, _, test_loader, _ = build_loaders(
        dataset_name, dataset_cache=cache, verbose=False
    )

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device, is_class)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'metrics': result['test_metrics'],
        'is_classification': is_class,
    }


def plot_roc_curve(y_true, y_prob, dataset_name, save_path):
    """Generate ROC curve from real predictions."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {dataset_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved ROC curve: {save_path}")
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, dataset_name, save_path, normalize=False):
    """Generate confusion matrix from real predictions."""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = ' (Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''

    plt.figure(figsize=(7, 6))

    # Plot
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)

    # Labels
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f'{cm[i, j]:{fmt}}'
            plt.text(j, i, value, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {dataset_name}{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved confusion matrix: {save_path}")


def plot_precision_recall_curve(y_true, y_prob, dataset_name, save_path):
    """Generate Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {ap:.3f})')
    plt.axhline(y=np.mean(y_true), color='navy', linestyle='--',
                label=f'Baseline ({np.mean(y_true):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {dataset_name}', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved PR curve: {save_path}")
    return ap


def main():
    """Generate all real evaluation figures."""

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    output_dir = Path("figures/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = Path("results/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Classification datasets
    classification_datasets = ['tox21', 'herg']

    all_results = {}

    for ds_name in classification_datasets:
        print(f"\n{'='*70}")
        print(f"Processing {ds_name}")
        print(f"{'='*70}")

        # Train and get predictions
        preds = train_and_get_predictions(ds_name, seed=42, device=device)

        y_true = preds['y_true']
        y_pred = preds['y_pred']
        y_prob = preds['y_prob']

        # Save predictions as .npz
        npz_path = predictions_dir / f"{ds_name}_predictions.npz"
        np.savez(npz_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        print(f"  Saved predictions: {npz_path}")

        # Generate figures
        display_name = "Tox21 (NR-AR)" if ds_name == 'tox21' else "hERG"

        # ROC curve
        roc_auc = plot_roc_curve(
            y_true, y_prob, display_name,
            output_dir / f"roc_curve_{ds_name}.png"
        )

        # Confusion matrix (raw)
        plot_confusion_matrix(
            y_true, y_pred, display_name,
            output_dir / f"confusion_matrix_{ds_name}.png",
            normalize=False
        )

        # Confusion matrix (normalized)
        plot_confusion_matrix(
            y_true, y_pred, display_name,
            output_dir / f"confusion_matrix_{ds_name}_normalized.png",
            normalize=True
        )

        # Precision-Recall curve
        ap = plot_precision_recall_curve(
            y_true, y_prob, display_name,
            output_dir / f"pr_curve_{ds_name}.png"
        )

        # Compute final metrics
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        all_results[ds_name] = {
            'auc_roc': roc_auc,
            'avg_precision': ap,
            'f1': f1,
            'accuracy': acc,
            'n_test': len(y_true),
            'n_positive': int(np.sum(y_true == 1)),
            'n_negative': int(np.sum(y_true == 0)),
        }

    # Save summary
    summary_path = output_dir / "classification_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY (from real test predictions)")
    print("="*70)
    print(f"{'Dataset':<15} {'AUC-ROC':<10} {'F1':<10} {'Accuracy':<10} {'AP':<10}")
    print("-"*55)
    for ds_name, r in all_results.items():
        print(f"{ds_name:<15} {r['auc_roc']:.4f}     {r['f1']:.4f}     {r['accuracy']:.4f}     {r['avg_precision']:.4f}")

    print("\n" + "="*70)
    print("Generated figures:")
    print("="*70)
    for ds_name in classification_datasets:
        print(f"  - figures/paper/roc_curve_{ds_name}.png")
        print(f"  - figures/paper/confusion_matrix_{ds_name}.png")
        print(f"  - figures/paper/confusion_matrix_{ds_name}_normalized.png")
        print(f"  - figures/paper/pr_curve_{ds_name}.png")

    return all_results


if __name__ == "__main__":
    main()
