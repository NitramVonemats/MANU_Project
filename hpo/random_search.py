import argparse
import numpy as np

from ._finalize import train_with_best_to_summary
from .problem import build_problem
from .space import bounds, decode_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="TDC ADME dataset name (e.g. Half_Life_Obach).")
    ap.add_argument("--batch-train", type=int, default=32, dest="batch_train")
    ap.add_argument("--batch-eval", type=int, default=64, dest="batch_eval")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-fraction", type=float, default=0.1, dest="val_fraction")
    ap.add_argument("--run-dir", default="runs", dest="run_dir")
    args = ap.parse_args()

    algo_name = "random"

    problem = build_problem(
        dataset=args.dataset,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        val_fraction=args.val_fraction,
    )

    rng = np.random.RandomState(args.seed)
    lb, ub = bounds()
    best_f, best_x = float("inf"), None
    for i in range(1, args.trials + 1):
        x = lb + rng.rand(len(lb)) * (ub - lb)
        f = problem._evaluate(x)
        if f < best_f:
            best_f, best_x = f, x
        print(f"[Random] trial {i}/{args.trials}: val_rmse={f:.4f}; best={best_f:.4f}")

    if best_x is None:
        raise RuntimeError("Random search did not evaluate any configurations.")

    best = decode_vector(np.array(best_x, float))
    out_path = train_with_best_to_summary(
        best=best,
        dataset=args.dataset,
        epochs=args.epochs,
        patience=args.patience,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        seed=args.seed,
        out_dir=args.run_dir,
        algo_name=algo_name,
        best_val_rmse_from_search=float(best_f),
        trials=args.trials,
        device=args.device,
        val_fraction=args.val_fraction,
        dataset_cache=problem.dataset_cache,
    )
    print(f"[HPO] Single-file summary saved → {out_path}")


if __name__ == "__main__":
    main()
