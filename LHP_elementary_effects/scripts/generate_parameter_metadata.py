#!/usr/bin/env python3
"""Extract VISIT parameter metadata for the elementary-effects workflow.

This script cross-references the v2 range CSV (index + min/max columns) with the
baseline parameter file (`parameter_LHP_backup.txt`). The output CSV lists one
row per perturbable parameter, including the default value used by the new
screening project.

Example:
    python scripts/generate_parameter_metadata.py \
        --range-csv ../LHP_parameter_perturbation_v2/docs/range_of_paras_v2.csv \
        --parameter-file ../CODE_LHP/INPUT/parameter_LHP_backup.txt \
        --output docs/parameter_metadata.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ParameterSpec:
    name: str
    line_index: int
    minimum: float
    maximum: float
    struct_field: str
    unit: str
    notes: str
    default_value: float

    @property
    def suggested_step(self) -> float:
        # Simple heuristic (10th of the span). Caller may override later.
        return (self.maximum - self.minimum) / 10.0 if self.maximum > self.minimum else 0.0


def _try_float(token: str) -> Optional[float]:
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _clean_first_column(text: str) -> Optional[tuple[int, str]]:
    if not text or ":" not in text:
        return None
    idx_token, name = text.split(":", 1)
    idx_token = idx_token.strip()
    name = name.strip()
    if not idx_token or not name:
        return None
    try:
        line_index = int(idx_token)
    except ValueError:
        return None
    return line_index, name


def load_parameter_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def extract_specs(range_csv: Path, parameter_lines: List[str]) -> List[ParameterSpec]:
    specs: List[ParameterSpec] = []
    with range_csv.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            first = (row[0] or "").strip()
            if not first or first.startswith("#"):
                continue
            header = _clean_first_column(first)
            if header is None:
                continue
            line_index, name = header
            if line_index >= len(parameter_lines):
                continue

            min_token = row[1].strip() if len(row) > 1 and row[1] else ""
            max_token = row[2].strip() if len(row) > 2 and row[2] else ""
            minimum = _try_float(min_token)
            maximum = _try_float(max_token)
            if minimum is None or maximum is None:
                # Skip entries lacking numeric bounds (fixed / derived parameters)
                continue

            raw_line = parameter_lines[line_index].strip()
            if not raw_line:
                continue
            parts = raw_line.split()
            try:
                default_value = float(parts[0])
            except (IndexError, ValueError):
                continue

            struct_field = row[4].strip() if len(row) > 4 and row[4] else ""
            unit = row[5].strip() if len(row) > 5 and row[5] else ""
            notes = row[6].strip() if len(row) > 6 and row[6] else ""

            specs.append(
                ParameterSpec(
                    name=name,
                    line_index=line_index,
                    minimum=minimum,
                    maximum=maximum,
                    struct_field=struct_field,
                    unit=unit,
                    notes=notes,
                    default_value=default_value,
                )
            )
    return specs


def write_metadata_csv(specs: List[ParameterSpec], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "line_index",
                "default_value",
                "minimum",
                "maximum",
                "suggested_step",
                "struct_field",
                "unit",
                "notes",
            ]
        )
        for spec in specs:
            writer.writerow(
                [
                    spec.name,
                    spec.line_index,
                    f"{spec.default_value:.8e}",
                    spec.minimum,
                    spec.maximum,
                    spec.suggested_step,
                    spec.struct_field,
                    spec.unit,
                    spec.notes,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate elementary-effects parameter metadata")
    parser.add_argument("--range-csv", required=True, help="Path to range_of_paras_v2.csv")
    parser.add_argument("--parameter-file", required=True, help="Path to parameter_LHP_backup.txt")
    parser.add_argument("--output", required=True, help="Destination CSV for metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    range_csv = Path(args.range_csv).resolve()
    parameter_file = Path(args.parameter_file).resolve()
    output_path = Path(args.output).resolve()

    if not range_csv.exists():
        raise FileNotFoundError(range_csv)
    if not parameter_file.exists():
        raise FileNotFoundError(parameter_file)

    parameter_lines = load_parameter_lines(parameter_file)
    specs = extract_specs(range_csv, parameter_lines)
    if not specs:
        raise RuntimeError("No perturbable parameters were found. Check the range CSV contents.")

    write_metadata_csv(specs, output_path)
    print(f"Wrote metadata for {len(specs)} parameters -> {output_path}")


if __name__ == "__main__":
    main()
