"""
Regenerate the GNN architecture-selection figures and supplementary tables
directly from the real screen results (results/architecture_selection/
architecture_selection.json). No hardcoded metrics.

Supersedes generate_gnn_architecture_comparison.py, whose values were hardcoded.

Outputs (to paper_final/images/ and results/architecture_selection/):
  - arch_comparison_all_datasets.png : per-dataset bar comparison (R2/RMSE/AUC)
  - arch_selection_summary.png       : average rank across regression datasets
  - table_S1_classification.tex      : LaTeX for the classification screen
  - table_S2_regression_ranking.tex  : LaTeX for the regression ranking
  - architecture_selection_summary.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "architecture_selection"
# Prefer the best-of-K validation-selected screen; fall back to the fixed-config one.
_SEARCH = OUT_DIR / "architecture_selection_search.json"
_FIXED = OUT_DIR / "architecture_selection.json"
DATA = _SEARCH if _SEARCH.exists() else _FIXED
IMG_DIR = PROJECT_ROOT / "paper_final" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

REG = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
CLF = ["tox21", "herg"]
REG_LABEL = {"Caco2_Wang": "Caco-2", "Half_Life_Obach": "Half-Life",
             "Clearance_Hepatocyte_AZ": "Clearance Hep.", "Clearance_Microsome_AZ": "Clearance Mic."}
CLF_LABEL = {"tox21": "Tox21 (NR-AR)", "herg": "hERG"}
# Display order for architectures
ARCH_ORDER = ["GraphConv", "GCN", "GAT", "GraphSAGE", "GIN", "TAG", "SGC", "Transformer"]


def load():
    with open(DATA) as f:
        return json.load(f)


def is_search(payload):
    return payload.get("config", {}).get("mode") == "search"


def fmt(mean, std):
    """Format a metric cell: 'x' for search (single value) else 'x$\\pm$s'."""
    if std is None or std == 0.0:
        return f"{mean:.3f}"
    return f"{mean:.3f}$\\pm${std:.3f}"


def arch_table(results, ds, metric):
    """Return {arch: (mean, std)} for a metric on a dataset."""
    out = {}
    for arch, d in results[ds]["architectures"].items():
        m, s = d.get(f"{metric}_mean"), d.get(f"{metric}_std")
        if m is not None:
            out[arch] = (m, s if s is not None else 0.0)
    return out


def fig_comparison(payload):
    results = payload["results"]
    present_reg = [d for d in REG if d in results]
    present_clf = [d for d in CLF if d in results]
    panels = present_reg + present_clf
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow))
    axes = np.array(axes).reshape(-1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ARCH_ORDER)))
    cmap = {a: colors[i] for i, a in enumerate(ARCH_ORDER)}

    for ax, ds in zip(axes, panels):
        is_clf = ds in present_clf
        metric = "auc_roc" if is_clf else "r2"
        tbl = arch_table(results, ds, metric)
        archs = [a for a in ARCH_ORDER if a in tbl]
        means = [tbl[a][0] for a in archs]
        stds = [tbl[a][1] for a in archs]
        bars = ax.bar(range(len(archs)), means, yerr=stds, capsize=3,
                      color=[cmap[a] for a in archs], edgecolor="black", linewidth=0.5,
                      error_kw=dict(ecolor="0.3", lw=1))
        best = int(np.argmax(means))  # higher R2/AUC is better
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)
        label = CLF_LABEL[ds] if is_clf else REG_LABEL[ds]
        unit = "Test AUC-ROC" if is_clf else "Test R$^2$"
        ax.set_title(f"{label}\n({unit} $\\uparrow$)", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(unit, fontsize=9)
        if not is_clf:
            ax.axhline(0, color="gray", lw=0.8)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    subtitle = ("best of {} validation-selected configurations".format(
                    len(payload["config"]["config_grid"])) if is_search(payload)
                else "mean $\\pm$ std over {} seeds".format(len(payload["config"]["seeds"])))
    fig.suptitle(f"GNN architecture screen across ADMET endpoints ({subtitle})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = IMG_DIR / "arch_comparison_all_datasets.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def compute_ranking(payload):
    """Average rank across regression datasets by mean R2 (higher better)."""
    results = payload["results"]
    present_reg = [d for d in REG if d in results]
    ranks = {a: [] for a in ARCH_ORDER}
    per_ds_r2 = {a: {} for a in ARCH_ORDER}
    params = {a: None for a in ARCH_ORDER}
    times = {a: [] for a in ARCH_ORDER}
    for ds in present_reg:
        tbl = arch_table(results, ds, "r2")
        # rank by R2 descending
        ordered = sorted(tbl.items(), key=lambda kv: kv[1][0], reverse=True)
        rank_of = {a: i + 1 for i, (a, _) in enumerate(ordered)}
        for a in tbl:
            ranks[a].append(rank_of[a])
            per_ds_r2[a][ds] = tbl[a][0]
            ent = results[ds]["architectures"][a]
            if ent.get("n_params_mean") is not None:
                params[a] = ent["n_params_mean"]
            if ent.get("train_time_mean") is not None:
                times[a].append(ent["train_time_mean"])
    summary = []
    for a in ARCH_ORDER:
        if not ranks[a]:
            continue
        summary.append({
            "Architecture": a,
            "Avg_Rank": float(np.mean(ranks[a])),
            "Mean_R2": float(np.mean(list(per_ds_r2[a].values()))),
            "Params_k": (params[a] / 1000.0) if params[a] else None,
            "Mean_time_s": float(np.mean(times[a])) if times[a] else None,
            **{REG_LABEL[ds]: per_ds_r2[a].get(ds) for ds in present_reg},
        })
    df = pd.DataFrame(summary).sort_values("Avg_Rank").reset_index(drop=True)
    return df, present_reg


def fig_selection_summary(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c44e52" if a == "GraphConv" else "#4c72b0" for a in df["Architecture"]]
    bars = ax.barh(df["Architecture"], df["Avg_Rank"], color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank across regression datasets (lower is better)")
    ax.set_title("GNN architecture selection summary\n"
                 "(average rank on ADME regression R$^2$; GraphConv = deployed backbone)",
                 fontweight="bold")
    for bar, r, p in zip(bars, df["Avg_Rank"], df["Params_k"]):
        txt = f"{r:.2f}" + (f"  ({p:.0f}k params)" if p is not None else "")
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=9)
    ax.legend(handles=[Patch(facecolor="#c44e52", label="GraphConv (deployed)"),
                       Patch(facecolor="#4c72b0", label="Other architectures")],
              loc="upper right")
    ax.set_xlim(0, df["Avg_Rank"].max() + 2.0)
    fig.tight_layout()
    out = IMG_DIR / "arch_selection_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def method_caption(payload):
    if is_search(payload):
        k = len(payload["config"]["config_grid"])
        return (f"each architecture is evaluated at its best of {k} configurations "
                "selected on the validation split (search over hidden dimension, "
                "number of layers, and learning rate)")
    n = len(payload["config"].get("seeds", []))
    return f"mean $\\pm$ std over {n} seeds under a single fixed configuration"


def latex_S1(payload):
    results = payload["results"]
    present_clf = [d for d in CLF if d in results]
    lines = [
        "% Auto-generated by scripts/make_architecture_figures.py -- do not edit by hand.",
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Preliminary classification architecture screen: " + method_caption(payload) +
        ". Test AUC-ROC and F1 on the scaffold-split test set. Best AUC per task in "
        "\\textbf{bold}.}",
        "\\label{tab:arch_classification}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "cc" * len(present_clf) + "}", "\\toprule",
        "\\textbf{Architecture} & " +
        " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{CLF_LABEL[d]}}}}}" for d in present_clf) + " \\\\",
        " & " + " & ".join("\\textbf{AUC} & \\textbf{F1}" for _ in present_clf) + " \\\\",
        "\\midrule",
    ]
    best_auc = {d: max(arch_table(results, d, "auc_roc").items(), key=lambda kv: kv[1][0])[0]
                for d in present_clf}
    for a in ARCH_ORDER:
        if a not in results[present_clf[0]]["architectures"]:
            continue
        cells = []
        for d in present_clf:
            auc = arch_table(results, d, "auc_roc")[a]
            f1 = arch_table(results, d, "f1")[a]
            aucs = fmt(*auc)
            if a == best_auc[d]:
                aucs = "\\textbf{" + aucs + "}"
            cells.append(aucs)
            cells.append(fmt(*f1))
        lines.append(f"{a} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = OUT_DIR / "table_S1_classification.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return "\n".join(lines)


def latex_S2(df, present_reg, payload):
    lines = [
        "% Auto-generated by scripts/make_architecture_figures.py -- do not edit by hand.",
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Regression architecture screen: average rank across the four ADME "
        "regression datasets (by test R$^2$, lower rank is better); " + method_caption(payload) +
        ". Per-dataset test R$^2$ shown for reference. GraphConv (deployed backbone) in "
        "\\textbf{bold}.}",
        "\\label{tab:arch_regression}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lc" + "c" * len(present_reg) + "cc}", "\\toprule",
        "\\textbf{Arch.} & \\textbf{Avg Rank} & " +
        " & ".join(f"\\textbf{{{REG_LABEL[d]}}}" for d in present_reg) +
        " & \\textbf{Params} & \\textbf{Time (s)} \\\\",
        "\\midrule",
    ]
    def num(v):  # proper LaTeX minus sign, matching the main manuscript
        return (f"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}")
    for _, row in df.iterrows():
        name = row["Architecture"]
        disp = f"\\textbf{{{name}}}" if name == "GraphConv" else name
        r2cells = " & ".join(num(row[REG_LABEL[d]]) for d in present_reg)
        pk = f"{row['Params_k']:.0f}k" if row["Params_k"] is not None else "--"
        tt = f"{row['Mean_time_s']:.1f}" if row["Mean_time_s"] is not None else "--"
        lines.append(f"{disp} & {row['Avg_Rank']:.2f} & {r2cells} & {pk} & {tt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = OUT_DIR / "table_S2_regression_ranking.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return "\n".join(lines)


def main():
    payload = load()
    fig_comparison(payload)
    df, present_reg = compute_ranking(payload)
    fig_selection_summary(df)
    df.to_csv(OUT_DIR / "architecture_selection_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'architecture_selection_summary.csv'}")
    s1 = latex_S1(payload)
    s2 = latex_S2(df, present_reg, payload)
    print("\n===== REGRESSION RANKING (real) =====")
    print(df.to_string(index=False))
    print("\n===== S1 =====\n" + s1)
    print("\n===== S2 =====\n" + s2)


if __name__ == "__main__":
    main()
