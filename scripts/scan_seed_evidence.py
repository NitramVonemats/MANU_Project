#!/usr/bin/env python3
"""Scan project CSV/JSON files for seed evidence.

This is a read-only audit helper. It reports files that contain explicit seed
columns/keys, files that mention literal 50, and whether any structured result
file contains 50 distinct random seeds.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules"}
SEED_KEYS = {"seed", "seeds", "random_seed", "random_state"}


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def collect_json_seeds(obj, seeds: set[int]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in SEED_KEYS:
                collect_seed_value(value, seeds)
            collect_json_seeds(value, seeds)
    elif isinstance(obj, list):
        for value in obj:
            collect_json_seeds(value, seeds)


def collect_seed_value(value, seeds: set[int]) -> None:
    if isinstance(value, list):
        for item in value:
            collect_seed_value(item, seeds)
    elif isinstance(value, int):
        seeds.add(value)
    elif isinstance(value, float) and value.is_integer():
        seeds.add(int(value))
    elif isinstance(value, str):
        for match in re.findall(r"\b\d+\b", value):
            seeds.add(int(match))


def scan_json(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    data = json.loads(text)
    seeds: set[int] = set()
    collect_json_seeds(data, seeds)
    return sorted(seeds), bool(re.search(r"\b50\b", text))


def scan_csv(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], [], bool(re.search(r"\b50\b", text))
        seed_cols = [col for col in reader.fieldnames if col and col.lower() in SEED_KEYS]
        seeds: set[int] = set()
        for row in reader:
            for col in seed_cols:
                value = (row.get(col) or "").strip()
                if re.fullmatch(r"\d+", value):
                    seeds.add(int(value))
    return seed_cols, sorted(seeds), bool(re.search(r"\b50\b", text) or "50" in path.name)


def main() -> int:
    reports = []
    errors = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or is_skipped(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                seeds, has_50 = scan_json(path)
                if seeds or has_50 or "seed" in path.name.lower():
                    reports.append(("JSON", rel, "", seeds, has_50))
            elif suffix == ".csv":
                seed_cols, seeds, has_50 = scan_csv(path)
                if seeds or has_50 or "seed" in path.name.lower():
                    reports.append(("CSV", rel, ",".join(seed_cols), seeds, has_50))
        except Exception as exc:
            errors.append((rel, str(exc)[:100]))

    reports.sort(key=lambda item: (-len(item[3]), item[1]))

    print("# Structured seed evidence in CSV/JSON")
    for kind, rel, cols, seeds, has_50 in reports:
        if seeds or has_50:
            col_text = f" seed_cols={cols}" if cols else ""
            print(f"{kind:4} {rel}{col_text} | seed_count={len(seeds)} seeds={seeds} | contains_50={has_50}")

    print("\n# Files with >=10 distinct seeds")
    many = [(rel, seeds) for _, rel, _, seeds, _ in reports if len(seeds) >= 10]
    if many:
        for rel, seeds in many:
            print(f"{rel}: {seeds}")
    else:
        print("NONE")

    print("\n# Files with >=50 distinct seeds")
    fifty = [(rel, seeds) for _, rel, _, seeds, _ in reports if len(seeds) >= 50]
    if fifty:
        for rel, seeds in fifty:
            print(f"{rel}: {seeds}")
    else:
        print("NONE")

    if errors:
        print("\n# Parse errors")
        for rel, err in errors[:50]:
            print(f"{rel}: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
