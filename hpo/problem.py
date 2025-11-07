from dataclasses import replace
from typing import Optional

import numpy as np
from niapy.problems import Problem

from GNN_test.optimized_gnn import (
    OptimizedGNNConfig,
    prepare_dataset,
    resolve_device,
    train_model,
)
from .space import bounds, decode_vector


class HyperParamProblem(Problem):
    """Niapy problem wrapper around the optimized molecular GNN."""

    def __init__(
        self,
        dataset_name: str,
        base_config: OptimizedGNNConfig,
        epochs: int,
        patience: int,
        seed: int,
        device: str = "auto",
        dataset_cache: Optional[dict] = None,
    ):
        self.dataset_name = dataset_name
        self.base_config = base_config
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = resolve_device(device)
        self.dataset_cache = dataset_cache or prepare_dataset(
            dataset_name=dataset_name,
            val_fraction=base_config.val_fraction,
            seed=seed,
            verbose=False,
        )

        lower, upper = bounds()
        super().__init__(dimension=len(lower), lower=lower, upper=upper)

    def _evaluate(self, x):
        hp = decode_vector(np.array(x, dtype=float))
        config = replace(self.base_config, **hp)

        try:
            result = train_model(
                dataset_name=self.dataset_name,
                config=config,
                epochs=self.epochs,
                patience=self.patience,
                device=self.device,
                seed=self.seed,
                dataset_cache=self.dataset_cache,
                evaluate_test=False,
                return_model=False,
                verbose=False,
            )
            return float(result["val_metrics"]["rmse"])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[HPO] Evaluation failed: {exc}")
            return float("inf")


def build_problem(
    dataset: str,
    batch_train: int,
    batch_eval: int,
    epochs: int,
    patience: int,
    seed: int,
    device: str = "auto",
    val_fraction: float = 0.1,
) -> HyperParamProblem:
    base_config = OptimizedGNNConfig(
        batch_train=batch_train,
        batch_eval=batch_eval,
        val_fraction=val_fraction,
    )
    return HyperParamProblem(
        dataset_name=dataset,
        base_config=base_config,
        epochs=epochs,
        patience=patience,
        seed=seed,
        device=device,
    )
