#!/usr/bin/env python3
"""
Regenerate the paper's data-driven figures with a single, consistent,
professional color palette (see viz_palette.py).

Data sources are the SAME as the paper's tables:
  * 01_algorithm_performance      <- runs/*/hpo_*.json   (= Table VIII)
  * gnn_vs_foundation_comparison  <- Table IV (model comparison)
  * foundation_ranking            <- Table IV
  * gnn_architecture_comparison   <- Tables V (regression) + VI (classification)
  * gnn_architecture_selection    <- Table VII
  * confusion_matrices            <- paper-authoritative confusion cells
  * multi_seed_validation         <- Table XI (mean +/- 95% CI; per-seed points
                                      were never saved, so we show mean +/- CI)

No training/HPO is run. Output goes straight to paper_final/images/.
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_palette import (  # noqa: E402
    DATASET_COLORS, DATASET_SHORT, MODEL_COLORS, ALGORITHM_COLORS, ARCH_COLORS,
    STABILITY_COLORS, SEQUENTIAL_CMAP, RANK_CMAP, BEST_EDGE, NEUTRAL_BASELINE,
    apply_style, color_for_dataset,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "paper_final", "images")
os.makedirs(OUT, exist_ok=True)

apply_style()

REG_DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
ALGOS = ["PSO", "ABC", "GA", "SA", "HC", "Random"]
MODELS = ["GNN", "ChemBERTa", "Morgan-FP", "MolE-FP", "MolCLR"]

# ---- Table IV: model comparison (R2 for regression, AUC for classification) ----
MODEL_R2 = {  # regression R^2
    "Caco2_Wang":              {"GNN": 0.481, "ChemBERTa": 0.478, "Morgan-FP": 0.200, "MolE-FP": 0.047,  "MolCLR": -0.079},
    "Half_Life_Obach":         {"GNN": 0.004, "ChemBERTa": -0.594, "Morgan-FP": -0.039, "MolE-FP": -0.329, "MolCLR": -0.025},
    "Clearance_Microsome_AZ":  {"GNN": 0.191, "ChemBERTa": 0.024, "Morgan-FP": 0.122, "MolE-FP": 0.059,  "MolCLR": -0.012},
    "Clearance_Hepatocyte_AZ": {"GNN": -1.019, "ChemBERTa": 0.029, "Morgan-FP": -0.015, "MolE-FP": 0.032, "MolCLR": -0.030},
}
MODEL_AUC = {  # classification AUC-ROC
    "herg":  {"GNN": 0.825, "ChemBERTa": 0.770, "Morgan-FP": 0.611, "MolE-FP": 0.672, "MolCLR": 0.504},
    "tox21": {"GNN": 0.742, "ChemBERTa": 0.728, "Morgan-FP": 0.722, "MolE-FP": 0.675, "MolCLR": 0.538},
}

# ---- Table V: best test R^2 per architecture (regression) ----
ARCH_ORDER = ["GraphConv", "GCN", "TAG", "GIN", "SGC", "Transformer", "GAT", "SAGE"]
ARCH_R2 = {
    "Half_Life_Obach":         {"GraphConv": 0.384, "GCN": 0.468, "TAG": 0.404, "GIN": 0.392, "SGC": 0.399, "Transformer": 0.327, "GAT": 0.370, "SAGE": 0.365},
    "Clearance_Hepatocyte_AZ": {"GraphConv": 0.087, "GCN": 0.041, "TAG": 0.027, "GIN": -0.126, "SGC": 0.009, "Transformer": -0.030, "GAT": -0.120, "SAGE": -0.072},
    "Clearance_Microsome_AZ":  {"GraphConv": 0.321, "GCN": 0.283, "TAG": 0.291, "GIN": 0.243, "SGC": 0.259, "Transformer": 0.149, "GAT": 0.147, "SAGE": 0.245},
}
# ---- Table VI: AUC per architecture (classification) ----
ARCH_AUC = {
    "tox21": {"GCN": 0.823, "GAT": 0.789, "GraphSAGE": 0.801},
    "herg":  {"GAT": 0.789, "GCN": 0.776, "GraphSAGE": 0.768},
}
# ---- Table VII: architecture selection summary ----
ARCH_RANK = [  # (arch, avg_rank, stability)
    ("GraphConv", 1.0, "High"), ("GCN", 2.7, "High"), ("TAG", 2.7, "Medium"),
    ("GIN", 4.0, "Medium"), ("Transformer", 4.0, "Medium"), ("SGC", 5.3, "High"),
    ("GAT", 7.0, "Low"), ("SAGE", 7.3, "Low"),
]
# ---- Table XI: multi-seed mean +/- 95% CI ----
MULTISEED = {  # dataset -> (metric, mean, ci_low, ci_high)
    "Caco2_Wang":              ("RMSE", 0.0033, 0.0027, 0.0039),
    "Half_Life_Obach":         ("RMSE", 20.05, 18.61, 21.50),
    "Clearance_Hepatocyte_AZ": ("RMSE", 52.37, 48.81, 55.93),
    "Clearance_Microsome_AZ":  ("RMSE", 53.46, 36.63, 70.30),
    "tox21":                   ("AUC", 0.711, 0.696, 0.727),
    "herg":                    ("AUC", 0.805, 0.778, 0.832),
}
# ---- Confusion matrices (paper-authoritative; rows=actual, cols=pred) ----
CONFUSION = {
    "tox21": {"title": "Tox21 (SA — best GNN)", "cm": [[1375, 7], [48, 23]], "acc": 0.962, "f1": 0.455, "auc": 0.742},
    "herg":  {"title": "hERG (ABC — best GNN)", "cm": [[23, 12], [23, 74]], "acc": 0.735, "f1": 0.809, "auc": 0.825},
}


def _short(ds):
    return DATASET_SHORT.get(ds, ds)


def _grouped_bars(ax, datasets, series_names, values, color_fn, ylabel, title,
                  baseline=None, ylim=None, highlight_best=None):
    """Generic grouped bar chart. values[series][dataset]."""
    x = np.arange(len(datasets))
    n = len(series_names)
    width = 0.8 / n
    for i, s in enumerate(series_names):
        offs = (i - n / 2 + 0.5) * width
        vals = [values[s].get(ds, np.nan) for ds in datasets]
        ax.bar(x + offs, vals, width, label=s, color=color_fn(s),
               edgecolor="white", linewidth=0.6, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, color="#444444", ls="--", lw=1, alpha=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(d) for d in datasets], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


# ===========================================================================
# 1. HPO algorithm performance (from runs/)  -> 01_algorithm_performance.png
# ===========================================================================
def load_runs_metric(metric):
    out = {a: {} for a in ALGOS}
    amap = {"PSO": "pso", "ABC": "abc", "GA": "ga", "SA": "sa", "HC": "hc", "Random": "random"}
    for ds in REG_DATASETS:
        for a in ALGOS:
            p = os.path.join(RUNS, ds, f"hpo_{ds}_{amap[a]}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                if metric == "train_time":
                    out[a][ds] = d.get("final_training", {}).get("train_time", np.nan)
                else:
                    out[a][ds] = d.get("final_training", {}).get("test_metrics", {}).get(metric, np.nan)
    return out


def fig_algorithm_performance():
    rmse = load_runs_metric("rmse")
    r2 = load_runs_metric("r2")
    ttime = load_runs_metric("train_time")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("HPO Algorithm Performance across ADME Datasets", fontsize=14, fontweight="bold")
    cfn = lambda s: ALGORITHM_COLORS[s]

    _grouped_bars(axes[0, 0], REG_DATASETS, ALGOS, rmse, cfn,
                  "Test RMSE (↓ better)", "(A) Test RMSE by Algorithm")
    _grouped_bars(axes[0, 1], REG_DATASETS, ALGOS, r2, cfn,
                  "Test R² (↑ better)", "(B) Test R² by Algorithm", baseline=0.0)
    _grouped_bars(axes[1, 0], REG_DATASETS, ALGOS, ttime, cfn,
                  "Training time (s)", "(C) Training Time by Algorithm")

    # Panel D: performance (RMSE) vs training time scatter
    axd = axes[1, 1]
    for a in ALGOS:
        xs = [ttime[a].get(ds, np.nan) for ds in REG_DATASETS]
        ys = [rmse[a].get(ds, np.nan) for ds in REG_DATASETS]
        axd.scatter(xs, ys, s=70, color=ALGORITHM_COLORS[a], label=a,
                    edgecolor="white", linewidth=0.8, zorder=3)
    axd.set_xlabel("Training time (s)")
    axd.set_ylabel("Test RMSE (↓ better)")
    axd.set_title("(D) Performance vs. Training Time")

    handles = [Patch(facecolor=ALGORITHM_COLORS[a], label=a) for a in ALGOS]
    axes[0, 1].legend(handles=handles, ncol=3, loc="upper right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "01_algorithm_performance.png")


# ===========================================================================
# 2. GNN vs foundation models (Table IV) -> gnn_vs_foundation_comparison.png
# ===========================================================================
def fig_gnn_vs_foundation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("GNN vs. Pretrained & Fingerprint Models", fontsize=14, fontweight="bold")
    cfn = lambda m: MODEL_COLORS[m]

    r2_series = {m: {ds: MODEL_R2[ds][m] for ds in REG_DATASETS} for m in MODELS}
    _grouped_bars(axes[0], REG_DATASETS, MODELS, r2_series, cfn,
                  "Test R² (↑ better)", "(A) ADME Regression — R²", baseline=0.0)

    tox = ["herg", "tox21"]
    auc_series = {m: {ds: MODEL_AUC[ds][m] for ds in tox} for m in MODELS}
    _grouped_bars(axes[1], tox, MODELS, auc_series, cfn,
                  "Test AUC-ROC (↑ better)", "(B) Toxicity Classification — AUC-ROC",
                  baseline=0.5, ylim=(0.4, 0.9))

    handles = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS]
    axes[0].legend(handles=handles, ncol=2, loc="lower left", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "gnn_vs_foundation_comparison.png")


# ===========================================================================
# 3. Foundation ranking heatmaps (Table IV)  -> foundation_ranking.png
# ===========================================================================
def _rank_panel(ax, datasets, value_map, title, higher_better=True):
    # value_map[dataset][model]; rank 1 = best per dataset
    ranks = np.zeros((len(MODELS), len(datasets)))
    for j, ds in enumerate(datasets):
        vals = [value_map[ds][m] for m in MODELS]
        order = np.argsort(vals)
        if higher_better:
            order = order[::-1]
        rk = {idx: r + 1 for r, idx in enumerate(order)}
        for i in range(len(MODELS)):
            ranks[i, j] = rk[i]
    im = ax.imshow(ranks, cmap=RANK_CMAP, vmin=1, vmax=len(MODELS), aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([_short(d) for d in datasets], fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=9)
    ax.set_title(title)
    for i in range(len(MODELS)):
        for j in range(len(datasets)):
            ax.text(j, i, f"{int(ranks[i, j])}", ha="center", va="center",
                    color="black", fontweight="bold", fontsize=10)
    ax.grid(False)
    return im


def fig_foundation_ranking():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [2, 1.1]})
    fig.suptitle("Model Rankings by Dataset (1 = Best, 5 = Worst)", fontsize=14, fontweight="bold")
    _rank_panel(axes[0], REG_DATASETS, MODEL_R2, "(A) ADME Regression — ranked by R²")
    im = _rank_panel(axes[1], ["herg", "tox21"], MODEL_AUC, "(B) Toxicity — ranked by AUC-ROC")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Rank (1 = best)")
    cbar.set_ticks(range(1, len(MODELS) + 1))
    _save(fig, "foundation_ranking.png", tight=False)


# ===========================================================================
# 4. GNN architecture comparison (Tables V + VI) -> gnn_architecture_comparison_all_datasets.png
# ===========================================================================
def _arch_bar_panel(ax, arch_vals, title, ylabel, baseline=None):
    archs = [a for a in ARCH_ORDER if a in arch_vals] + \
            [a for a in arch_vals if a not in ARCH_ORDER]
    vals = [arch_vals[a] for a in archs]
    best_i = int(np.argmax(vals))
    colors = [ARCH_COLORS.get(a, "#8C8C8C") for a in archs]
    bars = ax.bar(range(len(archs)), vals, color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)
    bars[best_i].set_edgecolor(BEST_EDGE)
    bars[best_i].set_linewidth(2.2)
    if baseline is not None:
        ax.axhline(baseline, color="#444444", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def fig_arch_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("GNN Architecture Comparison across Datasets", fontsize=14, fontweight="bold")
    reg = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
    for ax, ds in zip(axes[0], reg):
        _arch_bar_panel(ax, ARCH_R2[ds], f"{_short(ds)} — Best R²", "Best Test R² (↑)", baseline=0.0)
    cls = ["tox21", "herg"]
    for ax, ds in zip(axes[1], cls):
        _arch_bar_panel(ax, ARCH_AUC[ds], f"{_short(ds)} — AUC-ROC", "Test AUC-ROC (↑)", baseline=0.5)
        ax.set_ylim(0.5, 0.85)
    # last cell: legend / note
    axleg = axes[1, 2]
    axleg.axis("off")
    handles = [Patch(facecolor=ARCH_COLORS[a], label=a) for a in ARCH_ORDER]
    handles.append(Patch(facecolor="white", edgecolor=BEST_EDGE, linewidth=2.2, label="Best per dataset"))
    axleg.legend(handles=handles, loc="center", fontsize=10, title="Architecture",
                 title_fontproperties={"weight": "bold"})
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "gnn_architecture_comparison_all_datasets.png")


# ===========================================================================
# 5. Architecture selection summary (Table VII) -> gnn_architecture_selection_summary.png
# ===========================================================================
def fig_arch_selection():
    data = sorted(ARCH_RANK, key=lambda t: t[1], reverse=True)  # worst on top, best at bottom
    archs = [d[0] for d in data]
    ranks = [d[1] for d in data]
    stab = [d[2] for d in data]
    colors = [STABILITY_COLORS[s] for s in stab]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(archs))
    bars = ax.barh(y, ranks, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    # highlight selected backbone (GCN)
    for i, a in enumerate(archs):
        ax.text(ranks[i] + 0.08, y[i], f"{ranks[i]:.1f}", va="center", fontsize=9, color="#222")
        if a == "GCN":
            bars[i].set_edgecolor(BEST_EDGE)
            bars[i].set_linewidth(2.4)
            ax.annotate("Selected backbone", xy=(ranks[i], y[i]),
                        xytext=(ranks[i] + 1.6, y[i]), va="center", fontsize=9,
                        fontweight="bold", color=BEST_EDGE,
                        arrowprops=dict(arrowstyle="->", color=BEST_EDGE, lw=1.4))
    ax.set_yticks(y)
    ax.set_yticklabels(archs)
    ax.set_xlabel("Average rank across regression datasets (lower is better)")
    ax.set_title("GNN Architecture Selection Summary", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(ranks) + 2.5)
    ax.grid(axis="y", visible=False)
    handles = [Patch(facecolor=STABILITY_COLORS[s], label=f"{s} stability") for s in ["High", "Medium", "Low"]]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, "gnn_architecture_selection_summary.png")


# ===========================================================================
# 6. Confusion matrices -> confusion_matrices.png
# ===========================================================================
def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices for Best GNN Classification Models", fontsize=14, fontweight="bold")
    labels = ["Negative", "Positive"]
    for ax, key in zip(axes, ["tox21", "herg"]):
        info = CONFUSION[key]
        cm = np.array(info["cm"])
        base = color_for_dataset(key)
        im = ax.imshow(cm, cmap=SEQUENTIAL_CMAP)
        thr = cm.max() * 0.6
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white" if cm[i, j] > thr else "#222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{info['title']}\nAcc={info['acc']:.3f}  F1={info['f1']:.3f}  AUC={info['auc']:.3f}",
                     fontsize=10, color=base)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor(base); spine.set_linewidth(1.6)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "confusion_matrices.png")


# ===========================================================================
# 7. Multi-seed validation (Table XI: mean +/- 95% CI) -> multi_seed_boxplots.png
# ===========================================================================
def fig_multiseed():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Multi-Seed Validation — Mean ± 95% CI (n=5 seeds)", fontsize=14, fontweight="bold")
    order = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
             "Clearance_Microsome_AZ", "tox21", "herg"]
    for ax, ds in zip(axes.ravel(), order):
        metric, mean, lo, hi = MULTISEED[ds]
        c = DATASET_COLORS[ds]
        ax.bar([0], [mean], width=0.5, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar([0], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                    ecolor="#222222", elinewidth=1.6, capsize=8, capthick=1.6, zorder=4)
        ax.text(0, hi, f"  {mean:.4g}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks([0]); ax.set_xticklabels([_short(ds)])
        ax.set_ylabel(f"Test {metric}")
        ax.set_title(_short(ds), color=c)
        ax.set_xlim(-0.6, 0.6)
        pad = (hi - lo) * 0.6 + 1e-9
        ax.set_ylim(max(0, lo - pad), hi + pad * 1.4)
        ax.grid(axis="x", visible=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "multi_seed_validation.png")


def _save(fig, name, tight=True):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def main():
    print("Regenerating paper figures with unified palette ->", OUT)
    fig_algorithm_performance()
    fig_gnn_vs_foundation()
    fig_foundation_ranking()
    fig_arch_comparison()
    fig_arch_selection()
    fig_confusion()
    fig_multiseed()
    print("Done. (hpo_convergence_curves.png left unchanged: per-trial data not saved.)")


if __name__ == "__main__":
    main()
