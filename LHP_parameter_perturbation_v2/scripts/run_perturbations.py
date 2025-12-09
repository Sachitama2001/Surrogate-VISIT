"""Thin wrapper that reuses the v1 VISIT perturbation workflow.

The actual implementation lives in ../LHP_parameter_perturbation/scripts/run_perturbations.py
but we keep this entry point so that v2 can override it later if needed.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    legacy_script = base / "LHP_parameter_perturbation" / "scripts" / "run_perturbations.py"
    if not legacy_script.exists():
        raise FileNotFoundError(f"Legacy perturbation script not found: {legacy_script}")
    runpy.run_path(str(legacy_script), run_name="__main__")


if __name__ == "__main__":
    main()
