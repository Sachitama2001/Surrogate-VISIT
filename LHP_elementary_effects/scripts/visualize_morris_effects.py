#!/usr/bin/env python3
"""Visualize Morris elementary effects (mu_star) as bar charts.

Prerequisites: run `scripts/analyze_morris_effects.py` first.

Usage:
    python scripts/visualize_morris_effects.py \
        --config configs/elementary_effects_config.json \
        --effects-dir ../results/morris_effects \
        --output-dir ../results/morris_effects/plots \
        --top-n 15
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def resolve_path(base: Path, candidate: str) -> Path:
    ref = Path(candidate)
    return (base / ref).resolve() if not ref.is_absolute() else ref.resolve()


def load_effects(path: Path) -> List[Tuple[str, float, float, float, float]]:
    items: List[Tuple[str, float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                name = row["name"].strip()
                mu = float(row["mu"])
                mu_star = float(row["mu_star"])
                sigma = float(row["sigma"])
                mu_star_conf = float(row.get("mu_star_conf", 0.0) or 0.0)
            except (KeyError, ValueError, AttributeError):
                continue
            if not name:
                continue
            items.append((name, mu, mu_star, sigma, mu_star_conf))
    if not items:
        raise RuntimeError(f"No data found in {path}")
    return items


def plot_metric(metric: str, items: List[Tuple[str, float, float, float, float]], top_n: int, output_dir: Path) -> None:
    # Sort by mu_star descending
    items_sorted = sorted(items, key=lambda t: abs(t[2]), reverse=True)[:top_n]
    names = [t[0] for t in items_sorted]
    mu_stars = [t[2] for t in items_sorted]
    confs = [t[4] for t in items_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(names))
    ax.barh(y_pos, mu_stars, xerr=confs, color="#3b82f6", alpha=0.8, ecolor="#111827")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mu* (absolute mean elementary effect)")
    ax.set_title(f"Morris effects (top {top_n}) - {metric}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"morris_{metric.lower()}_top{top_n}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Morris effects")
    parser.add_argument("--config", default="configs/elementary_effects_config.json", help="Config JSON path")
    parser.add_argument("--effects-dir", default="../results/morris_effects", help="Directory containing morris_effects_<metric>.csv")
    parser.add_argument("--output-dir", default="../results/morris_effects/plots", help="Directory for output figures")
    parser.add_argument("--top-n", type=int, default=15, help="Number of parameters to display per metric")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    base = config_path.parent

    effects_dir = resolve_path(base, args.effects_dir)
    output_dir = resolve_path(base, args.output_dir)
    top_n = args.top_n

    if not effects_dir.exists():
        raise FileNotFoundError(effects_dir)

    metric_files: Dict[str, Path] = {}
    for path in effects_dir.glob("morris_effects_*.csv"):
        metric = path.stem.replace("morris_effects_", "").upper()
        metric_files[metric] = path

    if not metric_files:
        raise RuntimeError(f"No morris_effects_*.csv found in {effects_dir}")

    for metric, path in metric_files.items():
        items = load_effects(path)
        plot_metric(metric, items, top_n, output_dir)


if __name__ == "__main__":
    main()
