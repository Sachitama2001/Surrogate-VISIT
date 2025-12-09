#!/usr/bin/env python3
"""Rank parameters that drive variability (annual/daily SD) using Spearman correlation.

Reads the aggregated CSV (with parameters + mean_* + sd_annual_* + sd_daily_*),
computes Spearman rank correlation between each parameter and the SD metrics, and
produces:
  - CSV tables with correlations for each variability metric
  - Bar plots of top-k parameters by |correlation|

Example:
  conda run -n deepl python scripts/analyze_variability_drivers.py \
    --summary results/morris_summary_timeseries_metrics.csv \
    --outdir results/morris_effects
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze variability drivers via Spearman correlation")
    p.add_argument("--summary", default="results/morris_summary_timeseries_metrics.csv", help="Aggregated summary CSV")
    p.add_argument("--metrics", nargs="*", default=["GPP", "NEP", "ER", "NPP"], help="Metrics to analyze")
    p.add_argument("--outdir", default="results/morris_effects", help="Output directory for CSV/plots")
    p.add_argument("--top-k", type=int, default=15, help="Top-k parameters to plot by |rho|")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rank_correlations(df: pd.DataFrame, params: List[str], target: str) -> pd.DataFrame:
    rhos: List[Dict[str, float]] = []
    for p in params:
        series = df[[p, target]].dropna()
        if series.empty:
            continue
        rho = series[p].corr(series[target], method="spearman")
        rhos.append({"name": p, "rho": rho})
    out = pd.DataFrame(rhos)
    if not out.empty:
        out["abs_rho"] = out["rho"].abs()
        out = out.sort_values("abs_rho", ascending=False)
    return out


def plot_bar(df: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["name"], df["rho"], color="#984ea3")
    plt.xticks(rotation=75, ha="right")
    plt.title(title)
    plt.ylabel("Spearman rho")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    params = [c for c in df.columns if c not in {"sample_id"} and not c.startswith("mean_") and not c.startswith("sd_")]

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    for metric in args.metrics:
        for kind in ("annual", "daily"):
            target = f"sd_{kind}_{metric}"
            if target not in df.columns:
                print(f"Skip {metric} {kind}: column missing")
                continue
            ranked = rank_correlations(df, params, target)
            if ranked.empty:
                continue

            csv_path = outdir / f"variability_drivers_{kind}_{metric}.csv"
            ranked.to_csv(csv_path, index=False)

            top = ranked.head(args.top_k)
            plot_bar(
                top,
                f"Top {args.top_k} params for {metric} {kind} SD",
                plots_dir / f"variability_drivers_{kind}_{metric}.png",
            )

    print(f"Variability driver analyses written under {outdir}")


if __name__ == "__main__":
    main()
