"""
Evaluation metrics for regression tasks
"""
import numpy as np

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_metrics_np(y_true, y_pred):
    """
    Compute RMSE, MAE, and R² metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with rmse, mae, and r2 metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ybar = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - ybar) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {"rmse": rmse, "mae": mae, "r2": r2}


def spearman_metric(y_true, y_pred):
    """
    Compute Spearman correlation coefficient

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Spearman correlation coefficient
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 3:
        return 0.0

    if SCIPY_AVAILABLE:
        try:
            corr = spearmanr(y_true, y_pred).correlation
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    else:
        # Manual Spearman calculation
        r_true = y_true.argsort().argsort().astype(float)
        r_pred = y_pred.argsort().argsort().astype(float)
        r_true = (r_true - r_true.mean()) / (r_true.std() + 1e-12)
        r_pred = (r_pred - r_pred.mean()) / (r_pred.std() + 1e-12)
        return float((r_true * r_pred).mean())
