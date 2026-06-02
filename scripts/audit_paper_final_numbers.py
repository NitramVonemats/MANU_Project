#!/usr/bin/env python3
"""Audit paper_final/main.tex numbers against project artifacts.

This script is intentionally read-only. It cross-checks the numerical claims
that have backing data in datasets/, runs/, figures/, docs/, and archive/.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ok(name: str, expected, actual, tol: float = 5e-4) -> bool:
    if isinstance(expected, (int, float)):
        passed = abs(float(expected) - float(actual)) <= tol
    else:
        passed = expected == actual
    status = "OK" if passed else "MISMATCH"
    print(f"{status:9} {name:58} paper={expected!r} source={actual!r}")
    return passed


def csv_rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def json_obj(path: str):
    with (ROOT / path).open(encoding="utf-8") as f:
        return json.load(f)


def primary_metric(dataset: str, algo: str, metric: str) -> float:
    path = ROOT / "runs" / dataset / f"hpo_{dataset}_{algo}.json"
    data = json_obj(str(path.relative_to(ROOT)))
    return float(data["final_training"]["test_metrics"][metric])


def main() -> int:
    failures = 0

    print("\n# Table III: dataset rows and Used column")
    tdc_counts = {
        "datasets/adme/Caco2_Wang.csv": 910,
        "datasets/adme/Half_Life_Obach.csv": 667,
        "datasets/adme/Clearance_Hepatocyte_AZ.csv": 1213,
        "datasets/adme/Clearance_Microsome_AZ.csv": 1102,
        "datasets/toxicity/Tox21.csv": 7258,
        "datasets/toxicity/hERG.csv": 655,
    }
    for path, expected in tdc_counts.items():
        failures += not ok(path, expected, len(csv_rows(path)), 0)

    used = csv_rows("figures/paper/dataset_statistics.csv")
    expected_used = {
        "Caco2 Wang": 819,
        "Half-Life Obach": 601,
        "Clearance Hepatocyte": 1092,
        "Clearance Microsome": 992,
        "Tox21": 6533,
        "hERG": 590,
    }
    for row in used:
        failures += not ok(f"Used {row['Dataset']}", expected_used[row["Dataset"]], int(row["Size"]), 0)

    print("\n# Class balance claims")
    tox = csv_rows("datasets/toxicity/Tox21.csv")
    herg = csv_rows("datasets/toxicity/hERG.csv")
    tox_pos = sum(float(r["Y"]) == 1.0 for r in tox) / len(tox) * 100
    herg_pos = sum(float(r["Y"]) == 1.0 for r in herg) / len(herg) * 100
    failures += not ok("Tox21 positive rate (%)", 4.2, round(tox_pos, 1), 0.05)
    failures += not ok("hERG blocker rate (%)", 68.9, round(herg_pos, 1), 0.05)

    print("\n# Table VIII: HPO algorithm comparison")
    hpo = {
        ("Caco2_Wang", "rmse"): {"pso": 0.0031, "abc": 0.0029, "ga": 0.0031, "sa": 0.0029, "hc": 0.0030, "random": 0.0027},
        ("Half_Life_Obach", "rmse"): {"pso": 21.66, "abc": 21.66, "ga": 21.66, "sa": 23.70, "hc": 24.52, "random": 22.31},
        ("Clearance_Hepatocyte_AZ", "rmse"): {"pso": 70.21, "abc": 72.04, "ga": 71.34, "sa": 72.04, "hc": 72.04, "random": 68.22},
        ("Clearance_Microsome_AZ", "rmse"): {"pso": 42.76, "abc": 42.29, "ga": 42.29, "sa": 40.94, "hc": 41.63, "random": 38.75},
        ("tox21", "auc_roc"): {"pso": 0.692, "abc": 0.735, "ga": 0.735, "sa": 0.742, "hc": 0.652, "random": 0.713},
        ("herg", "auc_roc"): {"pso": 0.747, "abc": 0.825, "ga": 0.747, "sa": 0.802, "hc": 0.821, "random": 0.747},
    }
    for (dataset, metric), algos in hpo.items():
        for algo, expected in algos.items():
            actual = primary_metric(dataset, algo, metric)
            failures += not ok(f"{dataset} {algo} {metric}", expected, round(actual, 4 if actual < 1 else 2), 0.005)

    tpe_expected = {
        "Caco2_Wang": ("test_rmse_orig", 0.0029),
        "Half_Life_Obach": ("test_rmse_orig", 21.48),
        "Clearance_Hepatocyte_AZ": ("test_rmse_orig", 80.32),
        "Clearance_Microsome_AZ": ("test_rmse_orig", 40.89),
        "tox21": ("test_metric", 0.722),
        "herg": ("test_metric", 0.756),
    }
    for dataset, (key, expected) in tpe_expected.items():
        data = json_obj(f"archive/old_experiments/history/old_results/tpe_{dataset}_results.json")
        failures += not ok(f"{dataset} TPE {key}", expected, round(float(data[key]), 4 if expected < 1 else 2), 0.005)

    print("\n# Table X: HPO vs Random CI")
    for row in csv_rows("figures/paper-sources-2/hpo_ci_summary.csv"):
        algo = row["Algorithm"]
        expected = {
            "PSO": (2, 0, 4, -5.08, -10.21, 0.05, -0.70),
            "ABC": (1, 0, 5, -3.86, -6.90, -0.49, -0.89),
            "GA": (2, 0, 4, -3.98, -9.13, 0.92, -0.57),
            "SA": (1, 0, 5, -3.94, -6.01, -0.96, -1.09),
            "HC": (0, 0, 6, -6.96, -9.38, -4.09, -1.89),
        }[algo]
        actual = (
            int(row["Wins"]), int(row["Ties"]), int(row["Losses"]),
            round(float(row["MeanRelImprovementPct"]), 2),
            round(float(row["CI95_low_pct"]), 2),
            round(float(row["CI95_high_pct"]), 2),
            round(float(row["Cohens_dz"]), 2),
        )
        failures += not ok(f"CI summary {algo}", expected, actual)

    print("\n# Table XI: multi-seed values")
    ms_summary = {r["dataset"]: r for r in csv_rows("figures/paper-sources-2/multi_seed_summary_fixed.csv")}
    for dataset, expected in {
        "Caco2_Wang": (0.0026, 0.0001, 0.0026, 0.0027),
        "Half_Life_Obach": (20.72, 1.42, 19.48, 21.96),
        "Clearance_Hepatocyte_AZ": (49.87, 1.15, 48.86, 50.88),
        "Clearance_Microsome_AZ": (42.02, 3.36, 39.08, 44.97),
        "tox21": (0.716, 0.012, 0.706, 0.727),
        "herg": (0.804, 0.018, 0.789, 0.819),
    }.items():
        row = ms_summary[dataset]
        precision = 4 if dataset == "Caco2_Wang" else (3 if row["primary_metric"] == "AUC" else 2)
        actual = tuple(round(float(row[k]), precision) for k in ["mean", "std", "ci95_low", "ci95_high"])
        failures += not ok(f"multiseed summary {dataset}", expected, actual)

    ms_json = json_obj("figures/paper-sources-2/multi_seed_results_fixed.json")
    failures += not ok("multiseed json seeds", [42, 123, 456, 789, 1011], ms_json["seeds"])

    print("\n# Table VI / VII source availability")
    arch_rows = csv_rows("figures/paper-sources-2/gnn_classification_architecture_results.csv")
    arch = {(r["dataset"], r["model"]): r for r in arch_rows}
    for key, expected in {
        ("tox21", "GCN"): (0.823, 0.756),
        ("tox21", "GAT"): (0.789, 0.712),
        ("tox21", "GraphSAGE"): (0.801, 0.734),
        ("herg", "GAT"): (0.789, 0.712),
        ("herg", "GCN"): (0.776, 0.698),
        ("herg", "GraphSAGE"): (0.768, 0.689),
    }.items():
        row = arch[key]
        actual = (round(float(row["test_auc"]), 3), round(float(row["test_f1"]), 3))
        failures += not ok(f"arch classification {key[0]} {key[1]}", expected, actual)

    arch_script = (ROOT / "scripts/generate_gnn_architecture_comparison.py").read_text(encoding="utf-8")
    for needle in ["'Test_AUC': [0.823, 0.789, 0.801]", "'Test_F1': [0.756, 0.712, 0.734]",
                   "'Test_AUC': [0.789, 0.776, 0.768]", "'Avg_Rank': [2.3, 2.0, 2.3, 6.0, 3.7, 6.7, 7.0, 6.0]"]:
        failures += not ok(f"architecture source contains {needle[:30]}", True, needle in arch_script)

    print(f"\nTOTAL FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
