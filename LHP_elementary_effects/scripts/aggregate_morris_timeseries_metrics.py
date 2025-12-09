#!/usr/bin/env python3
"""Aggregate per-sample daily/annual metrics into a single CSV.

For each sample (identified by sample_id), this script:
- Reads parameter values from the Morris design CSV.
- Picks the latest annual and daily VISIT outputs in the sample run folder.
- Computes mean metrics (annual-average, same as existing summary),
  interannual variability (standard deviation across annual rows), and
  daily variability (standard deviation across all daily rows).
- Writes a consolidated table to `results/morris_summary_timeseries_metrics.csv`.

Run example:
    conda run -n deepl python scripts/aggregate_morris_timeseries_metrics.py \
        --config configs/elementary_effects_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ANNUAL_COLUMN_INDICES = {
    "GPP": 1,
    "NPP": 2,
    "NEP": 3,
    "ER": 14,
}

# daily columns: [year, doy, month, mday, 9 met vars, aCO2, GPP, NPP, ER, NEP, ...]
DAILY_COLUMN_INDICES = {
    "GPP": 14,
    "NPP": 15,
    "ER": 16,
    "NEP": 17,
}


def resolve_path(base: Path, candidate: str) -> Path:
    p = Path(candidate)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_design(path: Path) -> Tuple[List[str], Dict[int, Dict[str, float]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or reader.fieldnames[0] != "sample_id":
            raise ValueError("Design CSV must start with sample_id column")
        param_names = [c for c in reader.fieldnames if c != "sample_id"]
        samples: Dict[int, Dict[str, float]] = {}
        for row in reader:
            sid = int(float(row["sample_id"]))
            values: Dict[str, float] = {}
            for name in param_names:
                token = row.get(name)
                if token is None or token == "":
                    continue
                try:
                    values[name] = float(token)
                except ValueError:
                    continue
            samples[sid] = values
    return param_names, samples


def pick_latest(sample_dir: Path, pattern: str) -> Path | None:
    files = list(sample_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def parse_metrics_file(path: Path, column_map: Dict[str, int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    max_idx = max(column_map.values(), default=0)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) <= max_idx:
                continue
            entry: Dict[str, float] = {}
            for metric, col in column_map.items():
                try:
                    entry[metric] = float(tokens[col])
                except (ValueError, IndexError):
                    pass
            rows.append(entry)
    return rows


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def summarize_annual(path: Path, metrics: Iterable[str]) -> Dict[str, float]:
    rows = parse_metrics_file(path, ANNUAL_COLUMN_INDICES)
    result: Dict[str, float] = {}
    for metric in metrics:
        vals = [r[metric] for r in rows if metric in r]
        result[f"mean_{metric}"] = mean(vals)
        result[f"sd_annual_{metric}"] = stdev(vals)
    return result


def summarize_daily(path: Path, metrics: Iterable[str]) -> Dict[str, float]:
    rows = parse_metrics_file(path, DAILY_COLUMN_INDICES)
    result: Dict[str, float] = {}
    for metric in metrics:
        vals = [r[metric] for r in rows if metric in r]
        result[f"sd_daily_{metric}"] = stdev(vals)
    return result


def build_fieldnames(param_names: List[str], metrics: Iterable[str]) -> List[str]:
    metric_list = list(metrics)
    fields = ["sample_id", *param_names]
    for metric in metric_list:
        fields.append(f"mean_{metric}")
    for metric in metric_list:
        fields.append(f"sd_annual_{metric}")
    for metric in metric_list:
        fields.append(f"sd_daily_{metric}")
    return fields


def aggregate(config_path: Path) -> Path:
    cfg = load_config(config_path)
    base = config_path.parent

    design_csv = resolve_path(base, cfg["design_csv"])
    runs_dir = resolve_path(base, cfg.get("runs_output_dir", "../results/morris_runs"))
    output_csv = resolve_path(base, cfg.get("timeseries_summary_csv", "../results/morris_summary_timeseries_metrics.csv"))
    metrics = cfg.get("metrics", ["GPP", "NEP", "ER", "NPP"])

    param_names, design = read_design(design_csv)
    fieldnames = build_fieldnames(param_names, metrics)

    rows_out: List[Dict[str, float]] = []
    for sample_id in sorted(design.keys()):
        sample_dir = runs_dir / f"sample_{sample_id:04d}"
        annual_path = pick_latest(sample_dir, "LHP_*_annual*.txt")
        daily_path = pick_latest(sample_dir, "LHP_*_daily*.txt")

        summary: Dict[str, float] = {"sample_id": sample_id}
        # parameter values
        for name in param_names:
            summary[name] = design[sample_id].get(name, float("nan"))

        if annual_path:
            summary.update(summarize_annual(annual_path, metrics))
        if daily_path:
            summary.update(summarize_daily(daily_path, metrics))

        rows_out.append(summary)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    return output_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate daily/annual Morris metrics into one CSV")
    p.add_argument("--config", default="configs/elementary_effects_config.json", help="Path to config JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    output = aggregate(config_path)
    print(f"Aggregated metrics written to {output}")


if __name__ == "__main__":
    main()
