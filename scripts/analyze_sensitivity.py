"""Analyze the HPO sensitivity sweep: which hyperparameters dominate variance.

Reads results/hpo_sensitivity/hpo_sensitivity_<dataset>.json (config -> validation
score) and reports, per hyperparameter: (i) permutation importance from a random
forest surrogate fit on the logged trials, and (ii) Spearman rank correlation with
the validation score. Together these quantify the 'effective dimensionality' of the
search space -- the small set of hyperparameters that actually drive performance.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "results" / "hpo_sensitivity"
DIMS = ["hidden_dim", "num_layers", "lr", "weight_decay", "head1", "head2", "head3"]


def analyze(path):
    d = json.loads(Path(path).read_text())
    trials = d["trials"]
    if len(trials) < 15:
        return None
    X = np.array([[t[k] for k in DIMS] for t in trials], dtype=float)
    # log-scale the log-uniform dims so importance reflects the sampled geometry
    X[:, DIMS.index("lr")] = np.log10(X[:, DIMS.index("lr")])
    X[:, DIMS.index("weight_decay")] = np.log10(X[:, DIMS.index("weight_decay")])
    y = np.array([t["score"] for t in trials], dtype=float)

    rf = RandomForestRegressor(n_estimators=400, random_state=0, oob_score=True)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=50, random_state=0)
    imp = perm.importances_mean
    imp = imp / (imp.sum() if imp.sum() > 0 else 1.0)

    spearman = {}
    for i, k in enumerate(DIMS):
        rho, p = spearmanr(X[:, i], y)
        spearman[k] = (float(rho), float(p))

    order = np.argsort(imp)[::-1]
    print(f"\n=== {d['dataset']}  (n={len(trials)} valid trials, "
          f"RF OOB R^2={rf.oob_score_:.3f}) ===")
    print(f"{'hyperparameter':<16}{'perm.imp(%)':>12}{'Spearman rho':>14}{'p':>10}")
    cum = 0.0
    top2 = 0.0
    for j, i in enumerate(order):
        k = DIMS[i]
        rho, p = spearman[k]
        cum += imp[i]
        if j < 2:
            top2 += imp[i]
        print(f"{k:<16}{100*imp[i]:>11.1f}{rho:>14.3f}{p:>10.3f}")
    print(f"Top-2 dimensions explain {100*top2:.0f}% of surrogate importance; "
          f"score range over trials = [{y.min():.3f}, {y.max():.3f}]")
    return {"dataset": d["dataset"], "n": len(trials), "oob_r2": rf.oob_score_,
            "importance": {DIMS[i]: float(imp[i]) for i in range(len(DIMS))},
            "spearman": spearman, "top2_importance": float(top2)}


def main():
    out = {}
    for f in sorted(IN_DIR.glob("hpo_sensitivity_*.json")):
        r = analyze(f)
        if r:
            out[r["dataset"]] = r
    (IN_DIR / "sensitivity_analysis.json").write_text(json.dumps(out, indent=2))
    print("\nSaved", IN_DIR / "sensitivity_analysis.json")


if __name__ == "__main__":
    main()
