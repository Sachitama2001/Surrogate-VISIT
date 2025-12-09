#!/usr/bin/env python3
"""Generate a Morris (elementary effects) design for VISIT LHP parameters.

The script expects a JSON config (see `configs/elementary_effects_config.json`). It
loads the metadata CSV created by `generate_parameter_metadata.py`, keeps only the
parameters with finite bounds, and relies on SALib to build Morris trajectories.

Example:
    python scripts/prepare_morris_design.py --config configs/elementary_effects_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: Path) -> List[Tuple[str, float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        entries: List[Tuple[str, float, float]] = []
        for row in reader:
            try:
                name = row["name"].strip()
                minimum = float(row["minimum"])
                maximum = float(row["maximum"])
            except (KeyError, ValueError, AttributeError):
                continue
            if not name or maximum <= minimum:
                continue
            entries.append((name, minimum, maximum))
    if not entries:
        raise RuntimeError("Metadata CSV did not contain perturbable parameters")
    return entries


def ensure_salib():
    try:
        from SALib.sample import morris  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "SALib is required. Install via `pip install SALib` in the active environment."
        ) from exc
    return morris


def generate_design(entries: List[Tuple[str, float, float]], morris_cfg: Dict) -> tuple[List[List[float]], List[str]]:
    morris_module = ensure_salib()
    problem = {
        "num_vars": len(entries),
        "names": [name for name, *_ in entries],
        "bounds": [[mn, mx] for _, mn, mx in entries],
    }
    num_traj = int(morris_cfg.get("num_trajectories", 10))
    num_levels = int(morris_cfg.get("num_levels", 6))
    seed = morris_cfg.get("seed")
    optimal_traj = morris_cfg.get("optimal_trajectories")
    if optimal_traj is None and "grid_jump" in morris_cfg:
        # Backward compatibility with older configs
        optimal_traj = morris_cfg.get("grid_jump")
    optimal_traj = int(optimal_traj) if optimal_traj else None
    local_opt = bool(morris_cfg.get("local_optimization", True))
    sample = morris_module.sample(
        problem,
        N=num_traj,
        num_levels=num_levels,
        optimal_trajectories=optimal_traj,
        local_optimization=local_opt,
        seed=seed,
    )
    return sample.tolist(), problem["names"]


def write_design_csv(samples: List[List[float]], names: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *names])
        for idx, row in enumerate(samples):
            writer.writerow([idx, *row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Morris design samples")
    parser.add_argument("--config", required=True, help="Path to elementary-effects config JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = load_config(config_path)
    base_dir = config_path.parent
    metadata_path = (base_dir / cfg["metadata_csv"]).resolve() if not Path(cfg["metadata_csv"]).is_absolute() else Path(cfg["metadata_csv"]).resolve()
    design_path = (base_dir / cfg["design_csv"]).resolve() if not Path(cfg["design_csv"]).is_absolute() else Path(cfg["design_csv"]).resolve()
    morris_cfg = cfg.get("morris", {})

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_path}. Run generate_parameter_metadata.py first."
        )

    entries = load_metadata(metadata_path)
    samples, names = generate_design(entries, morris_cfg)
    write_design_csv(samples, names, design_path)
    print(
        f"Generated {len(samples)} Morris samples (variables={len(names)}) -> {design_path}"
    )


if __name__ == "__main__":
    main()
