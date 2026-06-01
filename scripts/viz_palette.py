#!/usr/bin/env python3
"""
Shared professional color palette for all paper figures.
=======================================================
Single source of truth so that the SAME entity (dataset, model, HPO
algorithm, GNN architecture) always gets the SAME color across every
visualization. Colors are drawn from one cohesive, muted, colorblind-aware
base palette (Seaborn "deep/muted" family) to keep figures professional and
avoid clashing hues.

Import in any figure script:
    from viz_palette import (DATASET_COLORS, MODEL_COLORS, ALGORITHM_COLORS,
                             ARCH_COLORS, STABILITY_COLORS, apply_style,
                             color_for_dataset, ...)
"""

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Base professional palette (muted, harmonized, colorblind-friendly)
# ---------------------------------------------------------------------------
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B3"
TEAL   = "#429EA6"
BROWN  = "#937860"
GRAY   = "#8C8C8C"
GOLD   = "#CCB974"
HILITE = "#2A9D5C"   # strong green used to flag "ours" / "selected"

NEUTRAL_BASELINE = "#7F7F7F"   # Random Search baseline
BEST_EDGE = "#B8860B"          # dark gold edge to highlight best bar

# ---------------------------------------------------------------------------
# DATASETS  (the user's example: Caco2 = the same blue everywhere)
# ---------------------------------------------------------------------------
DATASET_COLORS = {
    "Caco2_Wang":              BLUE,
    "Half_Life_Obach":         ORANGE,
    "Clearance_Hepatocyte_AZ": GREEN,
    "Clearance_Microsome_AZ":  RED,
    "tox21":                   PURPLE,
    "herg":                    TEAL,
}
# friendly aliases -> canonical dataset key
DATASET_ALIASES = {
    "caco2": "Caco2_Wang", "caco-2": "Caco2_Wang", "caco2_wang": "Caco2_Wang", "caco2 wang": "Caco2_Wang",
    "half_life": "Half_Life_Obach", "half-life": "Half_Life_Obach", "half_life_obach": "Half_Life_Obach",
    "clear._hep.": "Clearance_Hepatocyte_AZ", "clearance_hep.": "Clearance_Hepatocyte_AZ",
    "clearance hep.": "Clearance_Hepatocyte_AZ", "cl-hep": "Clearance_Hepatocyte_AZ",
    "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ",
    "clear._mic.": "Clearance_Microsome_AZ", "clearance_mic.": "Clearance_Microsome_AZ",
    "clearance mic.": "Clearance_Microsome_AZ", "cl-mic": "Clearance_Microsome_AZ",
    "clearance_microsome_az": "Clearance_Microsome_AZ",
    "tox21 (nr-ar)": "tox21", "tox21": "tox21",
    "herg": "herg", "herg (cardiotoxicity)": "herg",
}
DATASET_SHORT = {
    "Caco2_Wang": "Caco2", "Half_Life_Obach": "Half-Life",
    "Clearance_Hepatocyte_AZ": "Clear.-Hep", "Clearance_Microsome_AZ": "Clear.-Mic",
    "tox21": "Tox21", "herg": "hERG",
}

# ---------------------------------------------------------------------------
# MODEL FAMILIES (GNN highlighted as "ours")
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "GNN":       HILITE,
    "GNN-Best":  HILITE,
    "ChemBERTa": BLUE,
    "Morgan-FP": ORANGE,
    "MolE-FP":   PURPLE,
    "MolCLR":    GRAY,
}

# ---------------------------------------------------------------------------
# HPO ALGORITHMS (Random = neutral baseline)
# ---------------------------------------------------------------------------
ALGORITHM_COLORS = {
    "Random": NEUTRAL_BASELINE,
    "PSO":    BLUE,
    "ABC":    ORANGE,
    "GA":     GREEN,
    "SA":     RED,
    "HC":     PURPLE,
    "TPE":    TEAL,
}

# ---------------------------------------------------------------------------
# GNN ARCHITECTURES (GCN highlighted as the selected backbone)
# ---------------------------------------------------------------------------
ARCH_COLORS = {
    "GCN":        HILITE,
    "GraphConv":  BLUE,
    "GAT":        ORANGE,
    "GIN":        RED,
    "GraphSAGE":  PURPLE,
    "SAGE":       PURPLE,
    "TAG":        TEAL,
    "SGC":        BROWN,
    "Transformer": GRAY,
}

# ---------------------------------------------------------------------------
# STABILITY (semantic traffic-light, muted)
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
