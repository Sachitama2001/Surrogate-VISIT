"""Normalize parameter values from a CSV and plot true vs estimated scatter."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized parameter scatter")
    parser.add_argument("csv", type=Path, help="CSV file produced by evaluate_parameter_recovery.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: same folder with _norm_scatter.png)",
    )
    parser.add_argument(
        "--range-csv",
        type=Path,
        default=None,
        help="CSV listing parameter ranges (e.g., configs/range_of_paras.csv)",
    )
    parser.add_argument(
        "--include-prior",
        action="store_true",
        help="Include prior values when computing global fallback min/max",
    )
    return parser.parse_args()


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def load_bounds(range_csv: Path | None) -> dict[str, dict[str, float]]:
    if range_csv is None:
        return {}
    if not range_csv.exists():
        raise FileNotFoundError(range_csv)

    bounds: dict[str, dict[str, float]] = {}
    raw_lines = range_csv.read_text().splitlines()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.lower().startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        idx_name, min_str, max_str = parts[:3]
        name_match = re.search(r"\"([^\"]+)\"", idx_name)
        if name_match:
            name = name_match.group(1)
        else:
            name = idx_name.split(":", 1)[-1].strip().strip('\"')
        min_val = _parse_float(min_str)
        max_val = _parse_float(max_str)
        max_ref = None
        if max_val is None:
            max_ref = max_str.strip().strip('\"')
        bounds[name] = {"min": min_val, "max": max_val, "max_ref": max_ref}

    def resolve_max(param: str, stack: set[str]):
        entry = bounds[param]
        if entry.get("max") is not None or entry.get("max_ref") is None:
            return entry.get("max")
        ref = entry["max_ref"]
        if ref not in bounds:
            raise KeyError(f"max reference '{ref}' for parameter '{param}' not found in range file")
        if ref in stack:
            raise ValueError(f"Circular reference detected in range file: {' -> '.join(stack)} -> {ref}")
        stack.add(param)
        resolved = resolve_max(ref, stack)
        stack.remove(param)
        if resolved is None:
            raise ValueError(f"Unable to resolve max value for parameter '{param}' via reference '{ref}'")
        entry["max"] = resolved
        return resolved

    for name in bounds:
        resolve_max(name, set())

    return bounds


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if "true" not in df.columns or "estimated" not in df.columns:
        raise ValueError("CSV must contain 'true' and 'estimated' columns")

    bounds = load_bounds(args.range_csv)

    fallback_values = [df["true"].to_numpy(), df["estimated"].to_numpy()]
    if args.include_prior and "prior" in df.columns:
        fallback_values.append(df["prior"].to_numpy())
    fallback_all = np.concatenate(fallback_values)
    fallback_min = float(fallback_all.min())
    fallback_max = float(fallback_all.max())

    fallback_span = fallback_max - fallback_min if not np.isclose(fallback_max, fallback_min) else 1.0

    def normalize(val: float, name: str) -> float:
        entry = bounds.get(name)
        if not entry:
            return np.clip((val - fallback_min) / fallback_span, 0.0, 1.0)
        min_v = entry.get("min")
        max_v = entry.get("max")
        if min_v is None or max_v is None or np.isclose(max_v, min_v):
            return np.clip((val - fallback_min) / fallback_span, 0.0, 1.0)
        return np.clip((val - min_v) / (max_v - min_v), 0.0, 1.0)

    norm_true = []
    norm_est = []
    for _, row in df.iterrows():
        name = row["param_name"]
        norm_true.append(normalize(row["true"], name))
        norm_est.append(normalize(row["estimated"], name))
    norm_true = np.array(norm_true)
    norm_est = np.array(norm_est)
    abs_err = np.abs(df["estimated"] - df["true"])
    rel_err = abs_err / (np.abs(df["true"]) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(norm_true, norm_est, c=abs_err, cmap="viridis", s=60, edgecolor="k", alpha=0.85)
    lims = [0, 1]
    ax.plot(lims, lims, "r--", label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True (normalized)")
    ax.set_ylabel("Estimated (normalized)")
    ax.set_title("Normalized parameter scatter")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Abs error (raw units)")

    for i, name in enumerate(df["param_name"]):
        ax.annotate(name, (norm_true[i], norm_est[i]), textcoords="offset points", xytext=(3, 3), fontsize=7)

    fig.tight_layout()
    output_path = args.output or args.csv.with_name(args.csv.stem + "_norm_scatter.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved scatter plot to {output_path}")


if __name__ == "__main__":
    main()
