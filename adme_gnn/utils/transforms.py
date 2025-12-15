"""
Target transformation utilities for log-scale targets
"""
import numpy as np
from configs.base_config import TARGET_SCALER


def set_target_scaler(y_train):
    """
    Fit target scaler on training data (log transform + standardization)

    Args:
        y_train: Training labels
    """
    z = np.log(np.clip(y_train.astype(float), 1e-3, None))
    TARGET_SCALER.update({
        "mu": float(z.mean()),
        "sigma": float(z.std() or 1.0)
    })


def transform_y(y: float) -> float:
    """
    Transform target: log + standardize

    Args:
        y: Original target value

    Returns:
        Transformed target
    """
    return (np.log(max(1e-3, float(y))) - TARGET_SCALER["mu"]) / TARGET_SCALER["sigma"]


def inverse_y(t):
    """
    Inverse transform: unstandardize + exp

    Args:
        t: Transformed target value

    Returns:
        Original scale target
    """
    z = t * TARGET_SCALER["sigma"] + TARGET_SCALER["mu"]
    z = np.clip(z, -10, 10)
    return np.exp(z)
