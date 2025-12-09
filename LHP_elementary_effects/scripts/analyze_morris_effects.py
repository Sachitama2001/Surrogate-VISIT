#!/usr/bin/env python3
"""Compute Elementary Effects (Morris) statistics from run outputs.

Inputs (from config):
- metadata_csv: parameter names and bounds
- design_csv: Morris design with `sample_id` column
- summary_csv: run results with `sample_id` and mean_* metrics

Outputs:
- results/morris_effects/<metric>.csv : mu, mu_star, sigma for each parameter
- results/morris_effects/all_metrics.json : aggregate summary

Usage:
    python scripts/analyze_morris_effects.py --config configs/elementary_effects_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from SALib.analyze import morris


def resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: Path) -> Tuple[List[str], List[List[float]]]:
    names: List[str] = []
    bounds: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                name = row["name"].strip()
                mn = float(row["minimum"])
                mx = float(row["maximum"])
            except (KeyError, AttributeError, ValueError):
                continue
            if not name or mx <= mn:
                continue
            names.append(name)
            bounds.append([mn, mx])
    if not names:
        raise RuntimeError("No parameter metadata loaded")
    return names, bounds


def load_design(path: Path) -> Tuple[np.ndarray, List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in reader.fieldnames:
            raise ValueError("Design CSV missing sample_id column")
        rows: List[Tuple[int, List[float]]] = []
        for row in reader:
            try:
                sid = int(float(row["sample_id"]))
            except (TypeError, ValueError):
                continue
            values: List[float] = []
            for key in reader.fieldnames:
                if key == "sample_id":
                    continue
                token = row.get(key)
                if token is None or token == "":
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(token))
                    except ValueError:
                        values.append(np.nan)
            rows.append((sid, values))
    if not rows:
        raise RuntimeError("Design CSV had no data")
    rows.sort(key=lambda item: item[0])
    sample_ids = [sid for sid, _ in rows]
    matrix = np.array([vals for _, vals in rows], dtype=float)
    return matrix, sample_ids


def load_metric_vectors(summary_path: Path, metrics: List[str]) -> Tuple[Dict[str, np.ndarray], List[int]]:
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in reader.fieldnames:
            raise ValueError("summary CSV missing sample_id column")
        rows: List[Dict] = []
        for row in reader:
            try:
                sid = int(float(row["sample_id"]))
            except (TypeError, ValueError):
                continue
            if row.get("status") and row["status"].strip().lower() != "success":
                continue
            entry = {"sample_id": sid}
            for metric in metrics:
                key = f"mean_{metric}"
                if key in row and row[key] not in (None, ""):
                    try:
                        entry[metric] = float(row[key])
                    except ValueError:
                        pass
            rows.append(entry)
    if not rows:
        raise RuntimeError("No usable rows in summary CSV")
    rows.sort(key=lambda item: item["sample_id"])
    sample_ids = [r["sample_id"] for r in rows]
    vectors: Dict[str, np.ndarray] = {}
    for metric in metrics:
        values = [r.get(metric, np.nan) for r in rows]
        vectors[metric] = np.array(values, dtype=float)
    return vectors, sample_ids


def ensure_alignment(design_ids: List[int], result_ids: List[int]) -> None:
    if design_ids != result_ids:
        raise RuntimeError("Sample ordering mismatch between design and results")


def analyze_metric(problem: Dict, X: np.ndarray, y: np.ndarray, num_levels: int) -> Dict:
    if np.isnan(y).any():
        raise RuntimeError("Output vector contains NaN; check summary CSV")
    return morris.analyze(
        problem,
        X,
        y,
        num_levels=num_levels,
        conf_level=0.95,
        print_to_console=False,
    )


def write_metric_csv(metric: str, analysis: Dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "mu", "mu_star", "sigma", "mu_star_conf"])
        for name, mu, mu_star, sigma, mu_star_conf in zip(
            analysis["names"],
            analysis["mu"],
            analysis["mu_star"],
            analysis["sigma"],
            analysis["mu_star_conf"],
        ):
            writer.writerow([name, mu, mu_star, sigma, mu_star_conf])


def write_summary_json(all_results: Dict[str, Dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        metric: {
            "names": list(res["names"]),
            "mu": res["mu"].tolist(),
            "mu_star": res["mu_star"].tolist(),
            "sigma": res["sigma"].tolist(),
            "mu_star_conf": res["mu_star_conf"].tolist(),
        }
        for metric, res in all_results.items()
    }
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Morris effects from run results")
    parser.add_argument("--config", default="configs/elementary_effects_config.json", help="Path to config JSON")
    parser.add_argument("--output-dir", default="../results/morris_effects", help="Directory to write analysis outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = load_config(config_path)
    base = config_path.parent

    metadata_path = resolve_path(base, cfg["metadata_csv"])
    design_path = resolve_path(base, cfg["design_csv"])
    summary_path = resolve_path(base, cfg.get("summary_csv", "../results/morris_metrics_summary.csv"))
    metrics = cfg.get("metrics", ["GPP", "NEP", "ER", "NPP"])
    num_levels = int(cfg.get("morris", {}).get("num_levels", 6))

    names, bounds = load_metadata(metadata_path)
    X, design_ids = load_design(design_path)
    y_vectors, result_ids = load_metric_vectors(summary_path, metrics)
    ensure_alignment(design_ids, result_ids)

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    output_dir = resolve_path(base, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict] = {}
    for metric in metrics:
        print(f"Analyzing {metric} ...")
        analysis = analyze_metric(problem, X, y_vectors[metric], num_levels)
        write_metric_csv(metric, analysis, output_dir / f"morris_effects_{metric}.csv")
        all_results[metric] = analysis

    write_summary_json(all_results, output_dir / "morris_effects_all_metrics.json")
    print(f"Finished. Results saved under {output_dir}")


if __name__ == "__main__":
    main()
