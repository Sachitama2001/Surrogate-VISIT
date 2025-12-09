#!/usr/bin/env python3
"""Plot distributions of level and variability metrics from the timeseries summary CSV.

Inputs:
  --summary: CSV produced by aggregate_morris_timeseries_metrics.py
  --outdir : Directory to write PNG plots (default: results/morris_effects/plots)

Outputs (per metric GPP/NEP/ER/NPP):
  - hist_mean_<metric>.png           : Histogram of mean_<metric>
  - hist_sd_annual_<metric>.png      : Histogram of sd_annual_<metric>
  - hist_sd_daily_<metric>.png       : Histogram of sd_daily_<metric>
  - scatter_mean_vs_sdAnnual_<metric>.png : mean vs annual SD
  - scatter_sdDaily_vs_sdAnnual_<metric>.png : daily SD vs annual SD
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize timeseries metrics")
    p.add_argument("--summary", default="results/morris_summary_timeseries_metrics.csv", help="Path to aggregated summary CSV")
    p.add_argument("--outdir", default="results/morris_effects/plots", help="Directory to save plots")
    p.add_argument("--metrics", nargs="*", default=["GPP", "NEP", "ER", "NPP"], help="Metrics to plot")
    return p.parse_args()


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hist(series, title: str, xlabel: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna(), bins=30, color="#377eb8", alpha=0.8, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def scatter(x, y, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(5.5, 5))
    plt.scatter(x, y, s=14, alpha=0.7, color="#4daf4a", edgecolors="none")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = pd.read_csv(summary_path)
    metrics: List[str] = args.metrics

    for metric in metrics:
        mean_col = f"mean_{metric}"
        sd_ann_col = f"sd_annual_{metric}"
        sd_day_col = f"sd_daily_{metric}"
        if not {mean_col, sd_ann_col, sd_day_col}.issubset(df.columns):
            print(f"Skip {metric}: required columns missing")
            continue

        hist(df[mean_col], f"{metric} mean", mean_col, outdir / f"hist_mean_{metric.lower()}.png")
        hist(df[sd_ann_col], f"{metric} annual SD", sd_ann_col, outdir / f"hist_sd_annual_{metric.lower()}.png")
        hist(df[sd_day_col], f"{metric} daily SD", sd_day_col, outdir / f"hist_sd_daily_{metric.lower()}.png")

        scatter(
            df[mean_col],
            df[sd_ann_col],
            f"{metric}: mean vs annual SD",
            mean_col,
            sd_ann_col,
            outdir / f"scatter_mean_vs_sdAnnual_{metric.lower()}.png",
        )
        scatter(
            df[sd_day_col],
            df[sd_ann_col],
            f"{metric}: daily SD vs annual SD",
            sd_day_col,
            sd_ann_col,
            outdir / f"scatter_sdDaily_vs_sdAnnual_{metric.lower()}.png",
        )

    print(f"Plots written to {outdir}")


if __name__ == "__main__":
    main()
