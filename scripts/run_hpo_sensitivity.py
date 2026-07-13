"""Hyperparameter-sensitivity sweep to test the 'effective dimensionality' claim.

The per-trial hyperparameter/performance pairs of the original HPO runs were not
persisted (only best configs were stored), so this script runs a dedicated random
search over the full 7-dimensional Table II search space on representative learnable
endpoints, logging the validation metric for every sampled configuration. The logged
(config -> performance) pairs are then analyzed (surrogate feature importance +
rank correlation) to identify which hyperparameters actually dominate performance
variance -- turning the Bergstra & Bengio 'effective dimensionality' analogy into a
finding demonstrated on this study's own search space.

Outputs: results/hpo_sensitivity/hpo_sensitivity_<dataset>.json
"""
import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig, build_loaders, train_epoch, evaluate,
    evaluate_classification, resolve_device, is_classification_dataset,
)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_architecture_selection import ConfigurableGNN  # noqa: E402

HIDDEN = [64, 96, 128, 192, 256, 384, 512]
LAYERS = [3, 4, 5, 6, 7]
HEAD1 = [128, 192, 256, 384, 512]
HEAD2 = [64, 96, 128, 192, 256]
HEAD3 = [32, 48, 64, 96, 128]
OUT_DIR = PROJECT_ROOT / "results" / "hpo_sensitivity"


def sample_config(rng):
    h = [rng.choice(HEAD1), rng.choice(HEAD2), rng.choice(HEAD3)]
    h = sorted(h, reverse=True)
    return {
        "hidden_dim": rng.choice(HIDDEN),
        "num_layers": rng.choice(LAYERS),
        "lr": float(10 ** rng.uniform(-4, -2)),
        "weight_decay": float(10 ** rng.uniform(-6, -2)),
        "head1": h[0], "head2": h[1], "head3": h[2],
    }


def train_config(dataset_name, params, seed, epochs, patience, device, cache):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    is_clf = is_classification_dataset(dataset_name)
    cfg = OptimizedGNNConfig(hidden_dim=params["hidden_dim"], num_layers=params["num_layers"],
                             head_dims=(params["head1"], params["head2"], params["head3"]),
                             lr=params["lr"], weight_decay=params["weight_decay"])
    train_loader, val_loader, test_loader, (mu, sigma), cache = build_loaders(
        dataset_name=dataset_name, batch_train=cfg.batch_train, batch_eval=cfg.batch_eval,
        val_fraction=cfg.val_fraction, seed=seed, dataset_cache=cache,
        return_cache=True, verbose=False)
    sample = cache["train"][0]
    input_dim = int(sample.x.size(-1)); adme_dim = int(sample.adme_features.numel())
    dropout = 0.5 if dataset_name == "Caco2_Wang" else 0.0
    label_noise = 0.05 if (dataset_name == "Caco2_Wang" and not is_clf) else 0.0
    model = ConfigurableGNN(arch="GraphConv", input_dim=input_dim, hidden_dim=cfg.hidden_dim,
                            num_layers=cfg.num_layers, adme_dim=adme_dim,
                            head_dims=cfg.head_dims, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("-inf") if is_clf else float("inf")
    best_state, no_improve = None, 0
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, opt, device, cfg.max_grad_norm,
                    label_noise=label_noise, is_classification=is_clf)
        vm = (evaluate_classification(model, val_loader, device) if is_clf
              else evaluate(model, val_loader, device, mu, sigma))
        cur = vm["f1"] if is_clf else vm["rmse"]
        improved = cur > best if is_clf else cur < best
        if improved:
            best, best_state, no_improve = cur, deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_final = (evaluate_classification(model, val_loader, device) if is_clf
                 else evaluate(model, val_loader, device, mu, sigma))
    # Higher-is-better performance score: AUC for clf, R2 for regression.
    score = val_final["auc_roc"] if is_clf else val_final["r2"]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Caco2_Wang", "herg"])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        rng = random.Random(args.seed)
        cache = None
        rows = []
        t0 = time.time()
        for i in range(args.n):
            p = sample_config(rng)
            score = train_config(ds, p, args.seed, args.epochs, args.patience, device, cache)
            if np.isfinite(score):
                rows.append({**p, "score": score})
            if (i + 1) % 10 == 0:
                print(f"[{ds}] {i+1}/{args.n} valid={len(rows)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        out = OUT_DIR / f"hpo_sensitivity_{ds}.json"
        out.write_text(json.dumps({"dataset": ds, "n_valid": len(rows), "trials": rows}, indent=2))
        print(f"Saved {out} ({len(rows)} valid trials)", flush=True)


if __name__ == "__main__":
    main()
