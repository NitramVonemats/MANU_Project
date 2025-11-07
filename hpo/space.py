import numpy as np

# Search spaces for the optimized molecular GNN
HIDDEN_CHOICES = np.array([64, 96, 128, 192, 256, 384, 512], dtype=float)
LAYER_CHOICES = np.array([3, 4, 5, 6, 7], dtype=float)
HEAD1_CHOICES = np.array([128, 192, 256, 384, 512], dtype=float)
HEAD2_CHOICES = np.array([64, 96, 128, 192, 256], dtype=float)
HEAD3_CHOICES = np.array([32, 48, 64, 96, 128], dtype=float)
LOG_LR_BOUNDS = (-4.0, -2.0)        # 1e-4 → 1e-2
LOG_WD_BOUNDS = (-6.0, -2.0)        # 1e-6 → 1e-2


def nearest(arr: np.ndarray, value: float) -> int:
    """Return the closest discrete choice from an array."""
    return int(arr[(np.abs(arr - value)).argmin()])


def bounds(_: str | None = None):
    """Continuous bounds for the optimizer (model_name kept for backwards compat)."""
    lb = np.array(
        [
            HIDDEN_CHOICES.min(),
            LAYER_CHOICES.min(),
            HEAD1_CHOICES.min(),
            HEAD2_CHOICES.min(),
            HEAD3_CHOICES.min(),
            LOG_LR_BOUNDS[0],
            LOG_WD_BOUNDS[0],
        ],
        dtype=float,
    )
    ub = np.array(
        [
            HIDDEN_CHOICES.max(),
            LAYER_CHOICES.max(),
            HEAD1_CHOICES.max(),
            HEAD2_CHOICES.max(),
            HEAD3_CHOICES.max(),
            LOG_LR_BOUNDS[1],
            LOG_WD_BOUNDS[1],
        ],
        dtype=float,
    )
    return lb, ub


def decode_vector(x, _: str | None = None):
    """Decode a continuous vector into discrete GNN hyperparameters."""
    x = np.asarray(x, dtype=float)
    hidden_dim = nearest(HIDDEN_CHOICES, x[0])
    num_layers = nearest(LAYER_CHOICES, x[1])

    head_dims = (
        nearest(HEAD1_CHOICES, x[2]),
        nearest(HEAD2_CHOICES, x[3]),
        nearest(HEAD3_CHOICES, x[4]),
    )
    # Ensure monotonic decrease for stability (largest -> smallest)
    head_dims = tuple(sorted((int(h) for h in head_dims), reverse=True))

    log_lr = float(np.clip(x[5], *LOG_LR_BOUNDS))
    log_wd = float(np.clip(x[6], *LOG_WD_BOUNDS))

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "head_dims": head_dims,
        "lr": 10.0 ** log_lr,
        "weight_decay": 10.0 ** log_wd,
    }
