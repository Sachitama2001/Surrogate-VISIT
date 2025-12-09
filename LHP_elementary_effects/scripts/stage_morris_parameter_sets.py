#!/usr/bin/env python3
"""Stage VISIT parameter files for each Morris design sample.

This utility does not run VISIT. It prepares per-sample copies of
`parameter_LHP.txt` (based on the backup) with values specified by the
Morris design CSV. Each sample gets its own folder under `results/` so
that downstream runners can execute VISIT and collect metrics.

Example:
    python scripts/stage_morris_parameter_sets.py --config configs/elementary_effects_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(base: Path, candidate: str) -> Path:
    ref = Path(candidate)
    return (base / ref).resolve() if not ref.is_absolute() else ref.resolve()


def load_metadata(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, int] = {}
        for row in reader:
            try:
                name = row["name"].strip()
                line_index = int(row["line_index"])
            except (KeyError, ValueError, AttributeError):
                continue
            if name:
                mapping[name] = line_index
    if not mapping:
        raise RuntimeError("Metadata CSV is empty or lacks line_index information")
    return mapping


def load_parameter_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def save_parameter_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _extract_value(lines: List[str], idx: int) -> float:
    token = lines[idx].split()[0]
    return float(token)


def normalize_humus_fractions(
    sample: Dict[str, float],
    parameter_lines: List[str],
    metadata: Dict[str, int],
) -> Dict[str, float]:
    """Force humus fractions to sum to 1 by solving f_hm_p = 1 - a - i.

    If a/i are absent from the sampled row, fall back to the baseline values
    in the parameter file. f_hm_p from the sample is ignored; it is recomputed.
    """

    needs_normalize = any(key in sample for key in ("f_hm_a", "f_hm_i", "f_hm_p"))
    if not needs_normalize:
        return sample

    def get_or_baseline(name: str, default: float = 0.0) -> float:
        if name in sample:
            return sample[name]
        if name in metadata:
            idx = metadata[name]
            if 0 <= idx < len(parameter_lines):
                try:
                    return _extract_value(parameter_lines, idx)
                except Exception:
                    return default
        return default

    a_val = get_or_baseline("f_hm_a")
    i_val = get_or_baseline("f_hm_i")
    # Remainder goes to passive pool; clip to [0, 1]
    p_val = max(0.0, min(1.0, 1.0 - a_val - i_val))

    sample = sample.copy()
    sample["f_hm_a"] = a_val
    sample["f_hm_i"] = i_val
    sample["f_hm_p"] = p_val
    return sample


def apply_sample(lines: List[str], metadata: Dict[str, int], sample: Dict[str, float]) -> List[str]:
    updated = lines.copy()

    # Keep humus fractions consistent (f_hm_p = 1 - a - i)
    sample = normalize_humus_fractions(sample, lines, metadata)

    for name, value in sample.items():
        if name not in metadata:
            continue
        idx = metadata[name]
        if idx >= len(updated):
            continue
        parts = updated[idx].split()
        if not parts:
            continue
        parts[0] = f"{value:.8e}"
        updated[idx] = " ".join(parts)
    return updated


def read_design_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in reader.fieldnames:
            raise ValueError("Design CSV must include a 'sample_id' column")
        for row in reader:
            sample_id = row.get("sample_id")
            if sample_id is None:
                continue
            values: Dict[str, float] = {"sample_id": float(sample_id)}
            for key, token in row.items():
                if key == "sample_id" or token is None or token == "":
                    continue
                try:
                    values[key] = float(token)
                except ValueError:
                    pass
            rows.append(values)
    if not rows:
        raise RuntimeError("No rows found in design CSV")
    return rows


def stage_samples(config_path: Path) -> None:
    cfg = load_config(config_path)
    base = config_path.parent

    parameter_file = resolve_path(base, cfg["parameter_file"])
    metadata_csv = resolve_path(base, cfg["metadata_csv"])
    design_csv = resolve_path(base, cfg["design_csv"])

    results_dir = base / "../results/morris_samples"
    results_dir = results_dir.resolve()

    parameter_lines = load_parameter_lines(parameter_file)
    metadata = load_metadata(metadata_csv)
    design_rows = read_design_rows(design_csv)

    manifest = []
    for row in design_rows:
        sample_id = int(row["sample_id"])
        sample_values = {k: v for k, v in row.items() if k != "sample_id"}
        staged_lines = apply_sample(parameter_lines, metadata, sample_values)
        sample_dir = results_dir / f"sample_{sample_id:04d}"
        target_file = sample_dir / "parameter_LHP.txt"
        save_parameter_lines(target_file, staged_lines)
        manifest.append(
            {
                "sample_id": sample_id,
                "parameter_file": str(target_file),
                "num_params": len(sample_values),
            }
        )

    manifest_path = results_dir / "staging_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(
        f"Staged {len(design_rows)} samples into {results_dir}."
        f" Manifest saved to {manifest_path}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage parameter files for Morris samples")
    parser.add_argument("--config", required=True, help="Path to elementary-effects config JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    stage_samples(config_path)


if __name__ == "__main__":
    main()
