#!/usr/bin/env python3
"""
Statistical Significance Tests for HPO Algorithms
==================================================
Performs Wilcoxon signed-rank tests comparing each HPO algorithm
against Random Search baseline.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "hpo"
FIGURES_DIR = PROJECT_ROOT / "figures" / "paper"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define datasets (use normalized metric direction for fair comparison)
DATASETS = {
    "Caco2_Wang": {"type": "regression", "metric_key": "rmse", "minimize": True},
    "Half_Life_Obach": {"type": "regression", "metric_key": "rmse", "minimize": True},
    "Clearance_Hepatocyte_AZ": {"type": "regression", "metric_key": "rmse", "minimize": True},
    "Clearance_Microsome_AZ": {"type": "regression", "metric_key": "rmse", "minimize": True},
    "tox21": {"type": "classification", "metric_key": "f1", "minimize": False},
    "herg": {"type": "classification", "metric_key": "f1", "minimize": False},
}

ALGORITHMS = ["pso", "abc", "ga", "sa", "hc"]
ALGO_NAMES = {
    "pso": "PSO",
    "abc": "ABC",
    "ga": "GA",
    "sa": "SA",
    "hc": "HC",
    "random": "Random"
}


def load_all_results():
    """Load test metrics for all algorithm-dataset combinations."""
    results = {}

    for dataset, props in DATASETS.items():
        results[dataset] = {}

        for algo in ALGORITHMS + ["random"]:
            json_path = RESULTS_DIR / dataset / f"hpo_{dataset}_{algo}.json"

            if not json_path.exists():
                # Try lowercase
                json_path = RESULTS_DIR / dataset.lower() / f"hpo_{dataset.lower()}_{algo}.json"

            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)

                test_metrics = data.get("final_training", {}).get("test_metrics", {})
                metric_val = test_metrics.get(props["metric_key"])

                if metric_val is not None:
                    results[dataset][algo] = metric_val

    return results


def normalize_for_comparison(results):
    """
    Normalize metrics so that higher is always better.
    For RMSE (minimize), we use negative RMSE.
    For F1 (maximize), we keep as is.
    """
    normalized = {}

    for dataset, props in DATASETS.items():
        if dataset not in results:
            continue

        normalized[dataset] = {}
        for algo, val in results[dataset].items():
            if props["minimize"]:
                # Negative RMSE so higher is better
                normalized[dataset][algo] = -val
            else:
                normalized[dataset][algo] = val

    return normalized


def compute_win_tie_loss(results):
    """Compute win/tie/loss counts for each algorithm vs Random."""
    wtl = {algo: {"win": 0, "tie": 0, "loss": 0} for algo in ALGORITHMS}

    for dataset in DATASETS:
        if dataset not in results:
            continue
        if "random" not in results[dataset]:
            continue

        random_val = results[dataset]["random"]

        for algo in ALGORITHMS:
            if algo not in results[dataset]:
                continue

            algo_val = results[dataset][algo]

            # Higher is better (after normalization)
            if algo_val > random_val + 1e-9:
                wtl[algo]["win"] += 1
            elif algo_val < random_val - 1e-9:
                wtl[algo]["loss"] += 1
            else:
                wtl[algo]["tie"] += 1

    return wtl


def wilcoxon_test_vs_random(results):
    """
    Perform Wilcoxon signed-rank test for each algorithm vs Random.
    Returns p-values and effect sizes (rank-biserial correlation).
    """
    test_results = {}

    # Build paired arrays
    datasets_with_random = [ds for ds in DATASETS if ds in results and "random" in results[ds]]

    random_vals = np.array([results[ds]["random"] for ds in datasets_with_random])

    for algo in ALGORITHMS:
        algo_vals = []
        valid_random_vals = []

        for ds in datasets_with_random:
            if algo in results[ds]:
                algo_vals.append(results[ds][algo])
                valid_random_vals.append(results[ds]["random"])

        if len(algo_vals) < 3:
            test_results[algo] = {
                "n_pairs": len(algo_vals),
                "p_value": np.nan,
                "statistic": np.nan,
                "effect_size": np.nan,
                "interpretation": "Insufficient data"
            }
            continue

        algo_arr = np.array(algo_vals)
        random_arr = np.array(valid_random_vals)

        # Differences
        diff = algo_arr - random_arr

        # Skip if all differences are zero
        if np.allclose(diff, 0):
            test_results[algo] = {
                "n_pairs": len(algo_vals),
                "p_value": 1.0,
                "statistic": 0,
                "effect_size": 0,
                "interpretation": "No difference"
            }
            continue

        try:
            # Wilcoxon signed-rank test (two-sided)
            stat, p_val = stats.wilcoxon(algo_arr, random_arr, alternative='two-sided')

            # Effect size: rank-biserial correlation
            n = len(diff[diff != 0])
            r = 1 - (2 * stat) / (n * (n + 1))

            # Interpret effect size (Cohen's conventions adapted)
            if abs(r) < 0.1:
                effect_interp = "negligible"
            elif abs(r) < 0.3:
                effect_interp = "small"
            elif abs(r) < 0.5:
                effect_interp = "medium"
            else:
                effect_interp = "large"

            # Statistical significance
            if p_val < 0.05:
                if r > 0:
                    sig_interp = f"Significantly BETTER than Random (p={p_val:.4f}, {effect_interp} effect)"
                else:
                    sig_interp = f"Significantly WORSE than Random (p={p_val:.4f}, {effect_interp} effect)"
            else:
                sig_interp = f"No significant difference (p={p_val:.4f})"

            test_results[algo] = {
                "n_pairs": len(algo_vals),
                "p_value": p_val,
                "statistic": stat,
                "effect_size": r,
                "interpretation": sig_interp
            }

        except Exception as e:
            test_results[algo] = {
                "n_pairs": len(algo_vals),
                "p_value": np.nan,
                "statistic": np.nan,
                "effect_size": np.nan,
                "interpretation": f"Error: {str(e)}"
            }

    return test_results


def generate_latex_table(wilcoxon_results, wtl_results):
    """Generate LaTeX table for statistical significance results."""
    latex = r"""\begin{table}[t]
\centering
\caption{Statistical significance of HPO algorithms vs.\ Random Search (Wilcoxon signed-rank test, $n$=6 datasets).}
\label{tab:significance}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Algorithm} & \textbf{W/T/L} & \textbf{$p$-value} & \textbf{Effect ($r$)} & \textbf{Interpretation} \\
\midrule
"""

    for algo in ALGORITHMS:
        name = ALGO_NAMES[algo]
        w = wtl_results[algo]["win"]
        t = wtl_results[algo]["tie"]
        l = wtl_results[algo]["loss"]

        if algo in wilcoxon_results:
            p_val = wilcoxon_results[algo]["p_value"]
            r = wilcoxon_results[algo]["effect_size"]

            if np.isnan(p_val):
                p_str = "N/A"
                r_str = "N/A"
                interp = "Insufficient data"
            else:
                p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
                r_str = f"{r:.2f}"

                if p_val < 0.05:
                    interp = "Significant" if r > 0 else "Sig. worse"
                else:
                    interp = "Not significant"
        else:
            p_str = "N/A"
            r_str = "N/A"
            interp = "N/A"

        latex += f"{name} & {w}/{t}/{l} & {p_str} & {r_str} & {interp} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    print("=" * 60)
    print("Statistical Significance Analysis")
    print("Wilcoxon Signed-Rank Test: HPO Algorithms vs. Random Search")
    print("=" * 60)

    # Load and normalize results
    raw_results = load_all_results()
    normalized_results = normalize_for_comparison(raw_results)

    print("\n=== Raw Results (Test Metrics) ===")
    for ds in DATASETS:
        if ds not in raw_results:
            continue
        print(f"\n{ds}:")
        for algo in ALGORITHMS + ["random"]:
            if algo in raw_results[ds]:
                print(f"  {ALGO_NAMES.get(algo, algo):8s}: {raw_results[ds][algo]:.4f}")

    # Win/Tie/Loss
    wtl = compute_win_tie_loss(normalized_results)

    print("\n\n=== Win/Tie/Loss vs Random (higher = better) ===")
    for algo in ALGORITHMS:
        w, t, l = wtl[algo]["win"], wtl[algo]["tie"], wtl[algo]["loss"]
        print(f"{ALGO_NAMES[algo]:8s}: {w}W / {t}T / {l}L")

    # Wilcoxon tests
    wilcoxon = wilcoxon_test_vs_random(normalized_results)

    print("\n\n=== Wilcoxon Signed-Rank Test Results ===")
    for algo in ALGORITHMS:
        res = wilcoxon[algo]
        print(f"\n{ALGO_NAMES[algo]}:")
        print(f"  n_pairs:     {res['n_pairs']}")
        print(f"  p-value:     {res['p_value']:.4f}" if not np.isnan(res['p_value']) else f"  p-value:     N/A")
        print(f"  effect size: {res['effect_size']:.3f}" if not np.isnan(res['effect_size']) else f"  effect size: N/A")
        print(f"  {res['interpretation']}")

    # Generate LaTeX table
    latex_table = generate_latex_table(wilcoxon, wtl)

    output_path = FIGURES_DIR / "significance_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"\n\nLaTeX table saved: {output_path}")

    # Also save summary CSV
    summary_data = []
    for algo in ALGORITHMS:
        summary_data.append({
            "Algorithm": ALGO_NAMES[algo],
            "Wins": wtl[algo]["win"],
            "Ties": wtl[algo]["tie"],
            "Losses": wtl[algo]["loss"],
            "p_value": wilcoxon[algo]["p_value"],
            "effect_size": wilcoxon[algo]["effect_size"]
        })

    summary_df = pd.DataFrame(summary_data)
    csv_path = FIGURES_DIR / "statistical_significance.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")

    print("\n[OK] Statistical significance analysis complete!")


if __name__ == "__main__":
    main()
