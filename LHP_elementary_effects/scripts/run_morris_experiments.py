#!/usr/bin/env python3
"""Run VISIT LHP for each staged Morris sample and collect annual metrics."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

METRIC_COLUMN_INDICES = {
    "GPP": 1,  # ansis_ann[0]
    "NPP": 2,  # ansis_ann[1]
    "NEP": 3,  # ansis_ann[2]
    "ER": 14,  # ansis_ann[4]
}
OUTPUT_PATTERNS = [
    "LHP_*_annual*.txt",
    "LHP_*_daily*.txt",
    "LHP_*_spinup*.txt",
    "LHP_restart*.txt",
]


def resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(manifest_path: Path) -> List[Dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for entry in manifest:
        entry["sample_id"] = int(entry["sample_id"])
    return sorted(manifest, key=lambda item: item["sample_id"])


def compile_visit(code_dir: Path) -> None:
    print(f"Compiling VISIT in {code_dir} ...")
    subprocess.run(["make", "clean"], cwd=code_dir, check=False, capture_output=True, text=True)
    result = subprocess.run(["make"], cwd=code_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    if not (code_dir / "visitb").exists():
        raise RuntimeError("Compilation completed but visitb was not created")
    print("✓ Compilation successful")


def cleanup_previous_outputs(code_dir: Path) -> None:
    for pattern in OUTPUT_PATTERNS:
        for path in code_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def run_visit(code_dir: Path, log_path: Path, timeout: Optional[int]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        with subprocess.Popen(
            ["./visitb"],
            cwd=code_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:  # pragma: no cover - runtime guard
                process.kill()
                raise RuntimeError(f"VISIT timed out after {timeout}s") from exc
            if process.returncode != 0:
                raise RuntimeError(f"VISIT exited with code {process.returncode}")
    return time.time() - start


def collect_outputs(code_dir: Path, sample_dir: Path) -> List[Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    collected: List[Path] = []
    for pattern in OUTPUT_PATTERNS:
        for path in code_dir.glob(pattern):
            destination = sample_dir / path.name
            shutil.move(str(path), destination)
            collected.append(destination)
    return collected


def pick_annual_file(sample_dir: Path) -> Optional[Path]:
    annual_candidates = list(sample_dir.glob("LHP_*_annual*.txt"))
    if not annual_candidates:
        return None
    return max(annual_candidates, key=lambda item: item.stat().st_mtime)


def parse_annual_metrics(annual_path: Path, metrics: Iterable[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    needed_indices = {metric: METRIC_COLUMN_INDICES[metric] for metric in metrics}
    max_index = max(needed_indices.values(), default=0)

    with annual_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) <= max_index:
                continue
            try:
                year = int(float(tokens[0]))  # year column sometimes padded
            except ValueError:
                continue
            entry: Dict[str, float] = {"year": year}
            for metric, column_index in needed_indices.items():
                try:
                    entry[metric] = float(tokens[column_index])
                except (ValueError, IndexError):
                    continue
            rows.append(entry)
    return rows


def summarize_metrics(rows: List[Dict[str, float]], metrics: Iterable[str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for metric in metrics:
        values = [row[metric] for row in rows if metric in row]
        if values:
            summary[f"mean_{metric}"] = sum(values) / len(values)
    return summary


def write_summary_csv(summary_rows: List[Dict], destination: Path, metrics: Iterable[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [f"mean_{metric}" for metric in metrics]
    fieldnames = [
        "sample_id",
        "status",
        "elapsed_sec",
        "annual_file",
        "n_years",
        "notes",
        *metric_fields,
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def copy_parameter_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run_samples(
    manifest: List[Dict],
    metrics: List[str],
    live_param_file: Path,
    baseline_param: Path,
    code_dir: Path,
    runs_dir: Path,
    summary_csv: Path,
    start_id: int,
    end_id: Optional[int],
    max_samples: Optional[int],
    compile_once: bool,
    timeout: Optional[int],
) -> None:
    if compile_once:
        compile_visit(code_dir)

    summary_rows: List[Dict] = []
    selected = [entry for entry in manifest if entry["sample_id"] >= start_id]
    if end_id is not None:
        selected = [entry for entry in selected if entry["sample_id"] <= end_id]
    if max_samples is not None:
        selected = selected[:max_samples]

    if not selected:
        print("No samples matched the requested range.")
        return

    print(f"Running {len(selected)} samples (start={selected[0]['sample_id']}, end={selected[-1]['sample_id']})")

    try:
        for entry in selected:
            sample_id = entry["sample_id"]
            sample_dir = runs_dir / f"sample_{sample_id:04d}"
            log_path = sample_dir / "visit_log.txt"
            print("-" * 60)
            print(f"Sample {sample_id:04d}: using {entry['parameter_file']}")

            copy_parameter_file(Path(entry["parameter_file"]), live_param_file)
            cleanup_previous_outputs(code_dir)

            status = "success"
            notes = ""
            elapsed = 0.0
            try:
                elapsed = run_visit(code_dir, log_path, timeout)
                collected = collect_outputs(code_dir, sample_dir)
                if not collected:
                    status = "missing_outputs"
                    notes = "No LHP_* files produced"
                annual_file = pick_annual_file(sample_dir)
                if annual_file is None:
                    status = "missing_annual"
                    notes = "Annual output not found"
                    annual_rows: List[Dict[str, float]] = []
                else:
                    annual_rows = parse_annual_metrics(annual_file, metrics)
                    metric_summary = summarize_metrics(annual_rows, metrics)
                    if not annual_rows:
                        status = "empty_annual"
                        notes = "Annual file had no usable rows"
                    summary_rows.append(
                        {
                            "sample_id": sample_id,
                            "status": status,
                            "elapsed_sec": round(elapsed, 1),
                            "annual_file": str(annual_file.relative_to(runs_dir)),
                            "n_years": len(annual_rows),
                            "notes": notes,
                            **metric_summary,
                        }
                    )
                    continue
            except Exception as exc:  # pylint: disable=broad-except
                status = "failed"
                notes = str(exc)

            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "status": status,
                    "elapsed_sec": round(elapsed, 1),
                    "annual_file": "",
                    "n_years": 0,
                    "notes": notes,
                }
            )
    finally:
        copy_parameter_file(baseline_param, live_param_file)
        print("Restored baseline parameter_LHP.txt")

    write_summary_csv(summary_rows, summary_csv, metrics)
    print(f"Summary written to {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VISIT for staged Morris samples")
    parser.add_argument("--config", default="configs/elementary_effects_config.json", help="Path to config JSON")
    parser.add_argument("--start-id", type=int, default=0, help="Smallest sample_id to run")
    parser.add_argument("--end-id", type=int, help="Largest sample_id to run (inclusive)")
    parser.add_argument("--max-samples", type=int, help="Limit the number of samples to run")
    parser.add_argument("--no-compile", action="store_true", help="Skip recompiling visitb before runs")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds before treating a VISIT run as hung")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = load_config(config_path)
    config_dir = config_path.parent

    parameter_backup = resolve_path(config_dir, cfg["parameter_file"])
    metadata_csv = resolve_path(config_dir, cfg["metadata_csv"])
    design_csv = resolve_path(config_dir, cfg["design_csv"])
    _ = metadata_csv, design_csv  # keep parity with existing config keys

    manifest_path = resolve_path(config_dir, cfg.get("staging_manifest", "../results/morris_samples/staging_manifest.json"))
    runs_dir = resolve_path(config_dir, cfg.get("runs_output_dir", "../results/morris_runs"))
    summary_csv = resolve_path(config_dir, cfg.get("summary_csv", "../results/morris_metrics_summary.csv"))

    manifest = load_manifest(manifest_path)
    metrics = cfg.get("metrics", ["GPP", "NEP", "ER", "NPP"])
    for metric in metrics:
        if metric not in METRIC_COLUMN_INDICES:
            raise KeyError(f"Metric '{metric}' is not mapped to an annual column in ansis.c")

    input_dir = parameter_backup.parent
    live_param_file = input_dir / "parameter_LHP.txt"
    code_dir = input_dir.parent

    print("Config summary:")
    print(f"  parameter backup : {parameter_backup}")
    print(f"  live parameter   : {live_param_file}")
    print(f"  code directory   : {code_dir}")
    print(f"  manifest         : {manifest_path}")
    print(f"  runs directory   : {runs_dir}")
    print(f"  summary csv      : {summary_csv}")
    print(f"  metrics          : {', '.join(metrics)}")

    runs_dir.mkdir(parents=True, exist_ok=True)

    run_samples(
        manifest=manifest,
        metrics=list(metrics),
        live_param_file=live_param_file,
        baseline_param=parameter_backup,
        code_dir=code_dir,
        runs_dir=runs_dir,
        summary_csv=summary_csv,
        start_id=args.start_id,
        end_id=args.end_id,
        max_samples=args.max_samples,
        compile_once=not args.no_compile,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
