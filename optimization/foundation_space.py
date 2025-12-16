"""
Search space definitions for foundation model hyperparameter optimization.

Unlike GNN models which train end-to-end, foundation models use:
  1. Pre-trained encoder (or feature extractor)
  2. Downstream predictor (sklearn MLP)

This module defines the hyperparameter search space for:
  - Encoder parameters (projection dim, model-specific settings)
  - Predictor architecture (hidden layers, dimensions)
  - Training parameters (learning rate, weight decay, dropout)
"""
import numpy as np
from typing import Literal, Dict, Any

# ============================================================================
# SEARCH SPACE CHOICES
# ============================================================================

# Encoder projection dimension (output of encoder before predictor)
PROJ_DIM_CHOICES = np.array([128, 192, 256, 384, 512], dtype=float)

# Predictor hidden layer dimensions
HIDDEN1_CHOICES = np.array([128, 192, 256, 384, 512], dtype=float)
HIDDEN2_CHOICES = np.array([64, 96, 128, 192, 256], dtype=float)

# Learning rate and weight decay (log scale)
LOG_LR_BOUNDS = (-5.0, -2.0)  # 1e-5 → 1e-2
LOG_WD_BOUNDS = (-6.0, -2.0)  # 1e-6 → 1e-2

# Dropout
DROPOUT_BOUNDS = (0.0, 0.5)

# Model-specific parameters
# Morgan Fingerprint
MORGAN_NBITS_CHOICES = np.array([1024, 2048, 4096], dtype=float)
MORGAN_RADIUS_CHOICES = np.array([2, 3, 4], dtype=float)

# Transformer models (ChemBERTa, BioMed)
MAX_LENGTH_CHOICES = np.array([128, 256, 512], dtype=float)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def nearest(arr: np.ndarray, value: float) -> int:
    """Return the closest discrete choice from an array."""
    return int(arr[(np.abs(arr - value)).argmin()])


# ============================================================================
# BOUNDS FUNCTION
# ============================================================================

ModelType = Literal["morgan", "chemberta", "biomed", "molclr", "mole"]


def bounds(model_type: ModelType = "morgan"):
    """
    Return continuous bounds for the optimizer.
    
    Args:
        model_type: Type of foundation model
        
    Returns:
        (lower_bounds, upper_bounds) as numpy arrays
    """
    # Common parameters for all models
    common_lb = [
        PROJ_DIM_CHOICES.min(),
        HIDDEN1_CHOICES.min(),
        HIDDEN2_CHOICES.min(),
        LOG_LR_BOUNDS[0],
        LOG_WD_BOUNDS[0],
        DROPOUT_BOUNDS[0],
    ]
    common_ub = [
        PROJ_DIM_CHOICES.max(),
        HIDDEN1_CHOICES.max(),
        HIDDEN2_CHOICES.max(),
        LOG_LR_BOUNDS[1],
        LOG_WD_BOUNDS[1],
        DROPOUT_BOUNDS[1],
    ]
    
    # Model-specific parameters
    if model_type == "morgan":
        # Add: n_bits, radius
        lb = common_lb + [MORGAN_NBITS_CHOICES.min(), MORGAN_RADIUS_CHOICES.min()]
        ub = common_ub + [MORGAN_NBITS_CHOICES.max(), MORGAN_RADIUS_CHOICES.max()]
    elif model_type in ["chemberta", "biomed"]:
        # Add: max_length
        lb = common_lb + [MAX_LENGTH_CHOICES.min()]
        ub = common_ub + [MAX_LENGTH_CHOICES.max()]
    elif model_type in ["molclr", "mole"]:
        # No additional parameters for now
        lb = common_lb
        ub = common_ub
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return np.array(lb, dtype=float), np.array(ub, dtype=float)


# ============================================================================
# DECODE FUNCTION
# ============================================================================

def decode_vector(x: np.ndarray, model_type: ModelType = "morgan") -> Dict[str, Any]:
    """
    Decode a continuous vector into discrete hyperparameters.
    
    Args:
        x: Continuous vector from optimizer
        model_type: Type of foundation model
        
    Returns:
        Dictionary of hyperparameters
    """
    x = np.asarray(x, dtype=float)
    
    # Common parameters (indices 0-5)
    proj_dim = nearest(PROJ_DIM_CHOICES, x[0])
    hidden1 = nearest(HIDDEN1_CHOICES, x[1])
    hidden2 = nearest(HIDDEN2_CHOICES, x[2])
    
    # Ensure monotonic decrease (hidden1 >= hidden2)
    if hidden2 > hidden1:
        hidden1, hidden2 = hidden2, hidden1
    
    log_lr = float(np.clip(x[3], *LOG_LR_BOUNDS))
    log_wd = float(np.clip(x[4], *LOG_WD_BOUNDS))
    dropout = float(np.clip(x[5], *DROPOUT_BOUNDS))
    
    params = {
        "proj_dim": int(proj_dim),
        "hidden_dims": (int(hidden1), int(hidden2)),
        "lr": 10.0 ** log_lr,
        "weight_decay": 10.0 ** log_wd,
        "dropout": dropout,
    }
    
    # Model-specific parameters
    if model_type == "morgan":
        n_bits = nearest(MORGAN_NBITS_CHOICES, x[6])
        radius = nearest(MORGAN_RADIUS_CHOICES, x[7])
        params["n_bits"] = int(n_bits)
        params["radius"] = int(radius)
    elif model_type in ["chemberta", "biomed"]:
        max_length = nearest(MAX_LENGTH_CHOICES, x[6])
        params["max_length"] = int(max_length)
    elif model_type in ["molclr", "mole"]:
        # No additional parameters
        pass
    
    return params


# ============================================================================
# DIMENSION HELPER
# ============================================================================

def get_dimension(model_type: ModelType = "morgan") -> int:
    """Return the dimensionality of the search space for a given model type."""
    lb, ub = bounds(model_type)
    return len(lb)
