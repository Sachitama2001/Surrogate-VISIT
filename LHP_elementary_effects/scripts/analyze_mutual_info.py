#!/usr/bin/env python3
"""Compute mutual information between parameters and metrics.

Uses scikit-learn's mutual_info_regression to score each parameter against the
target metrics (default: mean_GPP/NEP/ER/NPP). Writes a CSV with MI scores and
prints top-k per metric.

Example:
    conda run -n deepl python scripts/analyze_mutual_info.py \
        --summary results/morris_summary_timeseries_metrics.csv \
        --outdir results/morris_effects --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mutual information between params and metrics")
    p.add_argument("--summary", default="results/morris_summary_timeseries_metrics.csv", help="Aggregated summary CSV")
    p.add_argument("--metrics", nargs="*", default=["mean_GPP", "mean_NEP", "mean_ER", "mean_NPP"], help="Target metric columns")
    p.add_argument("--outdir", default="results/morris_effects", help="Output directory for CSV")
    p.add_argument("--top-k", type=int, default=10, help="Top-k to print for each metric")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)

    # Parameter columns: exclude sample_id and known metric columns
    param_cols = [c for c in df.columns if c not in {"sample_id"} and not c.startswith("mean_") and not c.startswith("sd_")]

    X = df[param_cols].copy()
    # Handle NaNs by median fill per column
    X = X.apply(lambda s: s.fillna(s.median()))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "mutual_information_scores.csv"

    all_rows: List[dict] = []
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"skip {metric}: not in summary")
            continue
        y = df[metric].fillna(df[metric].median())
        mi = mutual_info_regression(X, y, random_state=0)
        rows = [
            {"metric": metric, "name": name, "mi": score}
            for name, score in zip(param_cols, mi)
        ]
        rows_sorted = sorted(rows, key=lambda r: r["mi"], reverse=True)
        all_rows.extend(rows_sorted)

        print(f"Top {args.top_k} (mutual information) for {metric}:")
        for rank, r in enumerate(rows_sorted[: args.top_k], 1):
            print(f"  {rank:2d}. {r['name']:20s} mi={r['mi']:.4f}")
        print()

    if all_rows:
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"Saved MI scores to {csv_path}")


if __name__ == "__main__":
    main()
