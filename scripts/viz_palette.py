#!/usr/bin/env python3
"""
Shared color palette for all paper figures -- single source of truth.
=====================================================================
Goals:
  * Professional, harmonious, colorblind-aware look. All colors come from one
    cohesive muted family (Tableau-10 style), so figures look consistent.
  * The SAME entity always gets the SAME color in every figure
    (e.g. Caco2 is always this blue; PSO is always this gold).
  * Datasets and HPO algorithms use DISJOINT colors -- no color ever means
    both a dataset and an algorithm (enforced by an assertion at import).

Because there are only ~10-12 genuinely distinct professional hues, the
model-family and architecture figures (each self-contained, with its own
legend, and never showing dataset/algorithm colors at the same time) reuse
colors from the same cohesive pool rather than introducing garish new hues.

Import in any figure script:
    from viz_palette import (DATASET_COLORS, MODEL_COLORS, ALGORITHM_COLORS,
                             ARCH_COLORS, STABILITY_COLORS, apply_style,
                             color_for_dataset, ...)
"""

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Cohesive professional base palette (Tableau-10 "modern", muted)
# ---------------------------------------------------------------------------
BLUE   = "#4E79A7"
ORANGE = "#F28E2B"
GREEN  = "#59A14F"
RED    = "#E15759"
PURPLE = "#B07AA1"
TEAL   = "#76B7B2"
GOLD   = "#EDC948"
BROWN  = "#9C755F"
ROSE   = "#D37295"
GRAY   = "#BAB0AC"
INDIGO = "#6B6FA8"
OLIVE  = "#8CA252"
DTEAL  = "#5BA3A3"
EMERALD = "#3C9D6E"   # "ours" highlight (GNN / selected backbone)

NEUTRAL_BASELINE = GRAY     # Random Search = neutral baseline
HILITE = EMERALD
BEST_EDGE = "#B8860B"       # dark-gold edge to flag the best/selected bar

# ---------------------------------------------------------------------------
# DATASETS (6) -- one fixed color per dataset, everywhere
# ---------------------------------------------------------------------------
DATASET_COLORS = {
    "Caco2_Wang":              BLUE,
    "Half_Life_Obach":         ORANGE,
    "Clearance_Hepatocyte_AZ": GREEN,
    "Clearance_Microsome_AZ":  RED,
    "tox21":                   PURPLE,
    "herg":                    TEAL,
}

# ---------------------------------------------------------------------------
# HPO ALGORITHMS (7) -- DISJOINT from the dataset palette
# ---------------------------------------------------------------------------
ALGORITHM_COLORS = {
    "Random": GRAY,
    "PSO":    GOLD,
    "ABC":    BROWN,
    "GA":     ROSE,
    "SA":     INDIGO,
    "HC":     OLIVE,
    "TPE":    DTEAL,
}

# ---------------------------------------------------------------------------
# MODEL FAMILIES (5) -- self-contained figure; GNN is the "ours" highlight.
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "GNN":       EMERALD,
    "GNN-Best":  EMERALD,
    "ChemBERTa": BLUE,
    "Morgan-FP": ORANGE,
    "MolE-FP":   PURPLE,
    "MolCLR":    GRAY,
}

# ---------------------------------------------------------------------------
# GNN ARCHITECTURES (8) -- self-contained figure; selected backbone (GraphConv)
# uses the "ours" emerald, and is also flagged with a gold border (BEST_EDGE).
# ---------------------------------------------------------------------------
ARCH_COLORS = {
    "GraphConv":   EMERALD,
    "GCN":         BLUE,
    "TAG":         ORANGE,
    "GIN":         RED,
    "SGC":         PURPLE,
    "Transformer": TEAL,
    "GAT":         GOLD,
    "SAGE":        BROWN,
    "GraphSAGE":   BROWN,   # alias of SAGE
}

# ---------------------------------------------------------------------------
# STABILITY (semantic traffic-light; its own legend, separate axis)
# ---------------------------------------------------------------------------
STABILITY_COLORS = {
    "High":   GREEN,
    "Medium": GOLD,
    "Low":    RED,
}

# Sequential colormap for count/intensity heatmaps (confusion matrices, etc.)
SEQUENTIAL_CMAP = "Blues"
# Diverging rank colormap (green = best rank)
RANK_CMAP = "RdYlGn_r"

# ---------------------------------------------------------------------------
# Enforce the key contract: datasets and algorithms never share a color.
# ---------------------------------------------------------------------------
_ds = set(DATASET_COLORS.values())
_al = set(ALGORITHM_COLORS.values())
assert not (_ds & _al), f"Dataset/algorithm color clash: {_ds & _al}"

# ---------------------------------------------------------------------------
# Short dataset labels and name aliases
# ---------------------------------------------------------------------------
DATASET_SHORT = {
    "Caco2_Wang": "Caco2", "Half_Life_Obach": "Half-Life",
    "Clearance_Hepatocyte_AZ": "Clear.-Hep", "Clearance_Microsome_AZ": "Clear.-Mic",
    "tox21": "Tox21", "herg": "hERG",
}
DATASET_ALIASES = {
    "caco2": "Caco2_Wang", "caco-2": "Caco2_Wang", "caco2_wang": "Caco2_Wang", "caco2 wang": "Caco2_Wang",
    "half_life": "Half_Life_Obach", "half-life": "Half_Life_Obach", "half_life_obach": "Half_Life_Obach",
    "clear._hep.": "Clearance_Hepatocyte_AZ", "clearance_hep.": "Clearance_Hepatocyte_AZ",
    "clearance hep.": "Clearance_Hepatocyte_AZ", "cl-hep": "Clearance_Hepatocyte_AZ",
    "cl_hepatocyte": "Clearance_Hepatocyte_AZ", "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ",
    "clear._mic.": "Clearance_Microsome_AZ", "clearance_mic.": "Clearance_Microsome_AZ",
    "clearance mic.": "Clearance_Microsome_AZ", "cl-mic": "Clearance_Microsome_AZ",
    "cl_microsome": "Clearance_Microsome_AZ", "clearance_microsome_az": "Clearance_Microsome_AZ",
    "tox21 (nr-ar)": "tox21", "tox21": "tox21",
    "herg": "herg", "herg (cardiotoxicity)": "herg",
}


def _norm(s):
    return str(s).strip().lower()


def color_for_dataset(name, default=GRAY):
    key = DATASET_ALIASES.get(_norm(name), name)
    return DATASET_COLORS.get(key, DATASET_COLORS.get(_norm(key), default))


def color_for_model(name, default=GRAY):
    return MODEL_COLORS.get(name, default)


def color_for_algorithm(name, default=GRAY):
    for k in ALGORITHM_COLORS:
        if k.lower() == _norm(name):
            return ALGORITHM_COLORS[k]
    return default


def color_for_arch(name, default=GRAY):
    for k in ARCH_COLORS:
        if k.lower() == _norm(name):
            return ARCH_COLORS[k]
    return default


def apply_style():
    """Apply a clean, professional matplotlib style for all figures."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
