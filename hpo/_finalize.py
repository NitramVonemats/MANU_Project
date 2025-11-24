import json
import os
from typing import Optional

from src.optimized_gnn import (
    OptimizedGNNConfig,
    prepare_dataset,
    resolve_device,
    train_model,
)


def _ensure_head_dims(head_dims) -> tuple[int, ...]:
    return tuple(int(h) for h in head_dims)


def train_with_best_to_summary(
    best: dict,
    dataset: str,
    epochs: int,
    patience: int,
    batch_train: int,
    batch_eval: int,
    seed: int,
    out_dir: str,
    algo_name: str,
    best_val_rmse_from_search: float,
    trials: int,
    device: str = "auto",
    val_fraction: float = 0.1,
    dataset_cache: Optional[dict] = None,
) -> str:
    """Train with best hyperparameters discovered during search and save a summary."""
    resolved_device = resolve_device(device)
    head_dims = best.get("head_dims", best.get("head_dim", (256, 128, 64)))

    config = OptimizedGNNConfig(
        hidden_dim=int(best["hidden_dim"]),
        num_layers=int(best["num_layers"]),
        head_dims=_ensure_head_dims(head_dims),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        batch_train=batch_train,
        batch_eval=batch_eval,
        val_fraction=val_fraction,
    )

    dataset_cache = dataset_cache or prepare_dataset(
        dataset_name=dataset,
        val_fraction=val_fraction,
        seed=seed,
        verbose=False,
    )

    result = train_model(
        dataset_name=dataset,
        config=config,
        epochs=epochs,
        patience=patience,
        device=resolved_device,
        seed=seed,
        dataset_cache=dataset_cache,
        evaluate_test=True,
        return_model=False,
        verbose=True,
    )

    out_dir_dataset = os.path.join(out_dir, dataset)
    os.makedirs(out_dir_dataset, exist_ok=True)
    out_path = os.path.join(out_dir_dataset, f"hpo_{dataset}_{algo_name}.json")

    payload = {
        "algo": algo_name,
        "dataset": dataset,
        "seed": seed,
        "search": {
            "trials": int(trials),
            "best_val_rmse": float(best_val_rmse_from_search),
            "best_params": {
                "hidden_dim": int(config.hidden_dim),
                "num_layers": int(config.num_layers),
                "head_dims": [int(h) for h in config.head_dims],
                "lr": float(config.lr),
                "weight_decay": float(config.weight_decay),
                "batch_train": int(config.batch_train),
                "batch_eval": int(config.batch_eval),
                "val_fraction": float(config.val_fraction),
            },
        },
        "final_training": {
            "epochs": int(epochs),
            "patience": int(patience),
            "train_time": float(result["train_time"]),
            "val_metrics": {k: float(v) for k, v in result["val_metrics"].items()},
            "test_metrics": (
                {k: float(v) for k, v in result["test_metrics"].items()}
                if result["test_metrics"] is not None
                else None
            ),
            "history": result["history"],
            "mu": float(result["mu"]),
            "sigma": float(result["sigma"]),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return os.path.abspath(out_path)
