"""Entry point for LHP parameter perturbations using the v2 range file.

This wrapper keeps the mature implementation from v1 but injects
v2-specific defaults (config + range CSV). The intent is to avoid
copying the 700+ line workflow while guaranteeing that LHS sampling
uses `docs/range_of_paras_v2.csv` under the v2 workspace.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = V2_ROOT.parent / "LHP_parameter_perturbation" / "scripts" / "run_perturbations.py"
DEFAULT_CONFIG = V2_ROOT / "configs" / "lhp_experiment_config.json"
DEFAULT_RANGE = V2_ROOT / "docs" / "range_of_paras_v2.csv"


def _ensure_flag(args: list[str], flag: str, value: Path) -> list[str]:
    """Append `flag value` when the user did not specify it explicitly."""

    if flag in args:
        return args
    for token in args:
        if token.startswith(f"{flag}="):
            return args
    return args + [flag, str(value)]


def main() -> None:
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(f"Legacy perturbation script not found: {LEGACY_SCRIPT}")
    if not DEFAULT_RANGE.exists():
        raise FileNotFoundError(f"range_of_paras_v2.csv not found: {DEFAULT_RANGE}")

    forwarded_args = sys.argv[1:]
    forwarded_args = _ensure_flag(forwarded_args, "--config", DEFAULT_CONFIG)
    forwarded_args = _ensure_flag(forwarded_args, "--range_file", DEFAULT_RANGE)

    sys.argv = [str(LEGACY_SCRIPT)] + forwarded_args
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
