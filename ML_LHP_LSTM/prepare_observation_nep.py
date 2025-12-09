#!/usr/bin/env python3
"""Generate processed NEP observation file from JapanFlux NEE data.

- Reads FLX_MY-LHP_JapanFLUX2024_ALLVARS_DD_2010-2020_1-3.csv
- Filters the target period (default: 2012-01-01 to 2016-12-31)
- Converts user-selected NEE column (e.g., NEE_vUT or NEE_PI_F, gC m^-2 day^-1)
    to NEP (Mg C ha^-1 day^-1) via NEP = -NEE/100
- Applies optional QC filtering and propagates an observation mask
- Writes a tidy CSV with date, NEP, QC (if available), and mask columns for downstream inversion
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

OBS_REL_PATH = "../LHP_observation/FLX_MY-LHP_JapanFLUX2024_ALLVARS_DD_2010-2020_1-3.csv"
DEFAULT_OUTPUT = "observations/nep_obs_2012_2016.csv"


@dataclass
class FilterStats:
    total_days: int
    selected_days: int
    valid_days: int

    @property
    def coverage(self) -> float:
        return self.valid_days / self.selected_days * 100 if self.selected_days else 0.0


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Observation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "TIMESTAMP" not in df.columns:
        raise ValueError("TIMESTAMP column missing from observation CSV")

    df["date"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d")
    return df


def extract_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


def compute_nep(
    df: pd.DataFrame,
    qc_threshold: float,
    nee_column: str,
    qc_column: str | None,
) -> Tuple[pd.DataFrame, FilterStats]:
    target = df.copy()

    if nee_column not in target.columns:
        raise ValueError(f"NEE column '{nee_column}' not found in observation CSV")

    nee = pd.to_numeric(target.get(nee_column), errors="coerce")

    qc = None
    if qc_column:
        if qc_column not in target.columns:
            raise ValueError(f"QC column '{qc_column}' not found in observation CSV")
        qc = pd.to_numeric(target.get(qc_column), errors="coerce")

    nee_invalid = nee.isna() | nee.isin([-9999, -9999.0])
    if qc is not None:
        qc_invalid = qc.isna() | (qc > qc_threshold)
    else:
        qc_invalid = pd.Series(False, index=target.index)

    invalid = nee_invalid | qc_invalid
    valid_mask = ~invalid

    nep = -nee / 100.0  # convert gC m^-2 day^-1 -> Mg C ha^-1 day^-1
    nep = nep.where(valid_mask)

    qc_values = qc.values if qc is not None else np.full(len(target), np.nan)

    output = pd.DataFrame(
        {
            "date": target["date"].values,
            "nee_gcm2": nee.values,
            "nee_source": nee_column,
            "nee_qc": qc_values,
            "nep_mg_ha": nep.values,
            "obs_mask": valid_mask.values,
        }
    )

    stats = FilterStats(
        total_days=len(df),
        selected_days=len(target),
        valid_days=int(valid_mask.sum()),
    )

    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare observation NEP CSV for inversion")
    parser.add_argument("--input", type=Path, default=Path(OBS_REL_PATH), help="Path to raw JapanFlux CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output CSV path (will be overwritten)",
    )
    parser.add_argument("--start", type=str, default="2012-01-01", help="Start date (inclusive)")
    parser.add_argument("--end", type=str, default="2016-12-31", help="End date (inclusive)")
    parser.add_argument(
        "--qc-threshold",
        type=float,
        default=0.5,
        help="Maximum acceptable QC value for valid observations",
    )
    parser.add_argument(
        "--nee-column",
        type=str,
        default="NEE_vUT",
        help="Source column used to compute NEP (e.g., NEE_vUT or NEE_PI_F)",
    )
    parser.add_argument(
        "--nee-qc-column",
        type=str,
        default="NEE_vUT_QC",
        help="QC column name. Pass an empty string to skip QC filtering.",
    )

    args = parser.parse_args()

    qc_column = args.nee_qc_column.strip() if args.nee_qc_column else None

    df_raw = load_raw_csv(args.input)
    df_period = extract_period(df_raw, args.start, args.end)
    processed, stats = compute_nep(
        df_period,
        qc_threshold=args.qc_threshold,
        nee_column=args.nee_column,
        qc_column=qc_column,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    print("Observation NEP file written:", output_path)
    print(f"NEE column: {args.nee_column} | QC column: {qc_column or 'None (skipped)'}")
    print(
        f"Selected days: {stats.selected_days} | Valid (mask=True): {stats.valid_days} "
        f"({stats.coverage:.1f}% coverage)"
    )
    print(
        "Columns: date, nee_gcm2, nee_source, nee_qc, nep_mg_ha, obs_mask\n"
        "Use obs_mask to filter days lacking reliable NEP"
    )


if __name__ == "__main__":
    main()
