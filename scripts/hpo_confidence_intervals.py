#!/usr/bin/env python3
"""
Confidence-Interval Analysis for HPO Algorithms vs. Random Search
=================================================================
Replaces the Wilcoxon p-value analysis with paired performance
differences and bootstrap 95% confidence intervals.

Uses ONLY existing saved HPO result files in runs/. No training is run.

For each optimizer vs Random Search and each dataset we compute a paired
improvement with a consistent "higher = better" direction:
    RMSE (regression):       improvement = RandomSearch_RMSE - Optimizer_RMSE
    F1 / AUC / R2 (class.):  improvement = Optimizer_metric - RandomSearch_metric

Because datasets differ by orders of magnitude in scale (RMSE 0.003 -> 68,
F1 0.43 -> 0.86), the cross-dataset aggregate uses RELATIVE improvement
(percent of the Random-Search baseline), which is unit-free and poolable.
"""
import json
import glob
import os
import numpy as np

RUNS_DIR = "runs"
OUT_DIR = os.path.join("figures", "paper-sources-2")
os.makedirs(OUT_DIR, exist_ok=True)

# Primary metric per dataset (matches the original significance analysis:
# RMSE for regression, F1 for the imbalanced classification tasks).
DATASETS = {
    "Caco2_Wang":             {"metric": "rmse", "minimize": True,  "label": "Caco2_Wang"},
    "Half_Life_Obach":        {"metric": "rmse", "minimize": True,  "label": "Half_Life"},
    "Clearance_Hepatocyte_AZ":{"metric": "rmse", "minimize": True,  "label": "Clear._Hep."},
    "Clearance_Microsome_AZ": {"metric": "rmse", "minimize": True,  "label": "Clear._Mic."},
    "tox21":                  {"metric": "f1",   "minimize": False, "label": "Tox21"},
    "herg":                   {"metric": "f1",   "minimize": False, "label": "hERG"},
}

ALGORITHMS = ["pso", "abc", "ga", "sa", "hc"]
ALGO_NAMES = {"pso": "PSO", "abc": "ABC", "ga": "GA", "sa": "SA", "hc": "HC", "random": "Random"}

N_BOOT = 10000
BOOT_SEED = 42


def load_metric(dataset, algo, metric):
    path = os.path.join(RUNS_DIR, dataset, f"hpo_{dataset}_{algo}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    val = d.get("final_training", {}).get("test_metrics", {}).get(metric)
    return val, path


def load_all():
    """raw[dataset][algo] = metric value."""
    raw = {}
    for ds, props in DATASETS.items():
        raw[ds] = {}
        for algo in ALGORITHMS + ["random"]:
            val, _ = load_metric(ds, algo, props["metric"])
            if val is not None:
                raw[ds][algo] = float(val)
    return raw


def improvement(random_val, algo_val, minimize):
    """Signed improvement, higher = better (optimizer beats Random)."""
    if minimize:                       # RMSE
        return random_val - algo_val
    return algo_val - random_val       # F1 / AUC / R2


def rel_improvement(random_val, algo_val, minimize):
    """Relative improvement as percent of the Random-Search baseline."""
    base = abs(random_val)
    if base == 0:
        return np.nan
    return 100.0 * improvement(random_val, algo_val, minimize) / base


def bootstrap_ci(diffs, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    """Percentile bootstrap CI for the mean of paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def cohens_dz(diffs):
    """Paired effect size: mean / std of the differences (d_z)."""
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diffs.mean() / sd)


def main():
    raw = load_all()

    print("=" * 78)
    print("Raw best-trial test metrics (per dataset / algorithm)")
    print("=" * 78)
    header = f"{'Dataset':24s} {'metric':6s} " + " ".join(f"{ALGO_NAMES[a]:>9s}" for a in ALGORITHMS + ["random"])
    print(header)
    for ds, props in DATASETS.items():
        row = f"{ds:24s} {props['metric']:6s} "
        row += " ".join(f"{raw[ds].get(a, float('nan')):9.4f}" for a in ALGORITHMS + ["random"])
        print(row)

    # Per-dataset improvements
    print("\n" + "=" * 78)
    print("Per-dataset paired improvement vs Random Search (positive = better)")
    print("=" * 78)
    per_dataset_rel = {algo: [] for algo in ALGORITHMS}
    per_dataset_abs = {algo: [] for algo in ALGORITHMS}
    detail_rows = []  # (algo, dataset, metric, random, algo_val, abs_imp, rel_imp, outcome)

    for algo in ALGORITHMS:
        print(f"\n--- {ALGO_NAMES[algo]} vs Random ---")
        for ds, props in DATASETS.items():
            if "random" not in raw[ds] or algo not in raw[ds]:
                continue
            r = raw[ds]["random"]
            a = raw[ds][algo]
            abs_imp = improvement(r, a, props["minimize"])
            rel_imp = rel_improvement(r, a, props["minimize"])
            per_dataset_abs[algo].append(abs_imp)
            per_dataset_rel[algo].append(rel_imp)
            outcome = "win " if abs_imp > 1e-9 else ("loss" if abs_imp < -1e-9 else "tie ")
            detail_rows.append((ALGO_NAMES[algo], props["label"], props["metric"], r, a, abs_imp, rel_imp, outcome.strip()))
            print(f"  {props['label']:14s} {props['metric']:5s}  random={r:10.4f}  algo={a:10.4f}  "
                  f"abs={abs_imp:+10.4f}  rel={rel_imp:+7.2f}%  [{outcome}]")

    # Summary table
    print("\n" + "=" * 78)
    print("Summary: optimizer vs Random Search  (n=6 datasets)")
    print("=" * 78)
    print(f"{'Algo':5s} {'W/T/L':8s} {'MeanRel%':>9s} {'95% CI (rel %)':>22s} {'d_z':>7s}")
    summary_rows = []
    for algo in ALGORITHMS:
        rels = np.array(per_dataset_rel[algo], dtype=float)
        w = int((rels > 1e-9).sum())
        l = int((rels < -1e-9).sum())
        t = int(len(rels) - w - l)
        mean_rel = float(rels.mean())
        lo, hi = bootstrap_ci(rels)
        dz = cohens_dz(rels)
        summary_rows.append((algo, w, t, l, mean_rel, lo, hi, dz))
        print(f"{ALGO_NAMES[algo]:5s} {f'{w}/{t}/{l}':8s} {mean_rel:8.2f}% "
              f"[{lo:+7.2f}%, {hi:+7.2f}%] {dz:7.2f}")

    # ---- Write LaTeX table ----
    latex = []
    latex.append(r"\begin{table}[!t]")
    latex.append(r"\centering")
    latex.append(r"\caption{HPO algorithms vs.\ Random Search: win/tie/loss, mean relative improvement, "
                 r"and bootstrap 95\% confidence intervals across $n$=6 datasets. "
                 r"Improvement direction is normalized so positive favors the optimizer "
                 r"(RMSE: Random$-$Algo; F1: Algo$-$Random), expressed as percent of the "
                 r"Random-Search baseline. CIs from 10{,}000 percentile bootstrap resamples; "
                 r"$d_z$ is the paired effect size.}")
    latex.append(r"\label{tab:ci_comparison}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Algorithm} & \textbf{W/T/L} & \textbf{Mean $\Delta$ (\%)} & "
                 r"\textbf{95\% CI (\%)} & \textbf{$d_z$} \\")
    latex.append(r"\midrule")
    for algo, w, t, l, mean_rel, lo, hi, dz in summary_rows:
        latex.append(f"{ALGO_NAMES[algo]} & {w}/{t}/{l} & {mean_rel:+.2f} & "
                     f"[{lo:+.2f}, {hi:+.2f}] & {dz:+.2f} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex_str = "\n".join(latex) + "\n"

    tex_path = os.path.join(OUT_DIR, "ci_comparison_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"\nLaTeX table written: {tex_path}")
    print("\n" + latex_str)

    # ---- Write CSVs (no pandas) ----
    sum_csv = os.path.join(OUT_DIR, "hpo_ci_summary.csv")
    with open(sum_csv, "w") as f:
        f.write("Algorithm,Wins,Ties,Losses,MeanRelImprovementPct,CI95_low_pct,CI95_high_pct,Cohens_dz\n")
        for algo, w, t, l, mean_rel, lo, hi, dz in summary_rows:
            f.write(f"{ALGO_NAMES[algo]},{w},{t},{l},{mean_rel:.4f},{lo:.4f},{hi:.4f},{dz:.4f}\n")
    print(f"Summary CSV written: {sum_csv}")

    det_csv = os.path.join(OUT_DIR, "hpo_paired_improvements.csv")
    with open(det_csv, "w") as f:
        f.write("Algorithm,Dataset,Metric,RandomValue,AlgoValue,AbsImprovement,RelImprovementPct,Outcome\n")
        for algo, ds, metric, r, a, abs_imp, rel_imp, outcome in detail_rows:
            f.write(f"{algo},{ds},{metric},{r:.6f},{a:.6f},{abs_imp:.6f},{rel_imp:.4f},{outcome}\n")
    print(f"Per-dataset CSV written: {det_csv}")

    print("\n[OK] Confidence-interval analysis complete.")


if __name__ == "__main__":
    main()
