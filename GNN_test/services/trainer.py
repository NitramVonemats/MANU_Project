"""
Training and evaluation utilities
"""
import time
import torch
import torch.nn as nn
from copy import deepcopy
from functional.metrics import compute_metrics_np, spearman_metric
from functional.transforms import inverse_y
from functional.utils import set_global_seed
from graph.loader import build_loaders


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Train model for one epoch

    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        device: Device (cpu or cuda)
        loss_fn: Loss function

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        y = data.y.view_as(pred)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, inverse=False):
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device
        inverse: Whether to inverse transform predictions

    Returns:
        Tuple of (metrics_dict, predictions, true_values)
    """
    model.eval()
    preds, trues = [], []

    for data in loader:
        data = data.to(device)
        p = model(data)
        preds.append(p.detach().cpu())
        trues.append(data.y.view_as(p).detach().cpu())

    if not preds:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan")
        }, [], []

    preds = torch.cat(preds).float().numpy()
    trues = torch.cat(trues).float().numpy()

    if inverse:
        preds = inverse_y(preds)
        trues = inverse_y(trues)

    metrics = compute_metrics_np(trues, preds)
    return metrics, preds, trues


def train_model(model_name, dataset_name, model, config, seed=42, epochs=150, patience=30, device=None):
    """
    Train a single model with early stopping

    Args:
        model_name: Name of the model (for logging)
        dataset_name: TDC dataset name
        model: PyTorch model
        config: Configuration dictionary
        seed: Random seed
        epochs: Maximum epochs
        patience: Early stopping patience
        device: Device (cpu or cuda)

    Returns:
        Dictionary with training results
    """
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[{model_name}] Training on {dataset_name}...")

    # Build data loaders
    train_loader, val_loader, test_loader, adme_dim = build_loaders(
        dataset_name,
        split_type=config['split_type'],
        batch_train=32,
        batch_eval=64
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    loss_fn = nn.MSELoss()

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    t0 = time.time()

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics, _, _ = evaluate(model, val_loader, device, inverse=True)

        monitor_metric = val_metrics["rmse"]

        if monitor_metric < best_metric:
            best_metric = monitor_metric
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1

        scheduler.step(monitor_metric)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch}: loss={train_loss:.4f}, val_R2={val_metrics['r2']:.3f}")

        if no_imp >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    val_metrics, val_pred, val_true = evaluate(model, val_loader, device, inverse=True)
    test_metrics, test_pred, test_true = evaluate(model, test_loader, device, inverse=True)
    val_spear = spearman_metric(val_true, val_pred)
    test_spear = spearman_metric(test_true, test_pred)

    print(f"  [{model_name}] FINAL: test_RMSE={test_metrics['rmse']:.3f}, test_R2={test_metrics['r2']:.3f}, Spear={test_spear:.3f}")

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_spearman": test_spear,
        "val_r2": val_metrics["r2"],
        "epochs_trained": best_epoch,
        "train_time_s": round(time.time() - t0, 2),
    }
