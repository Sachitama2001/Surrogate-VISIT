#!/usr/bin/env python3
"""Build observation-conditioned windows for parameter inversion.

The script stitches together:
- Dynamic inputs (meteorology + aCO2) from any VISIT perturbation sample
- Observation NEP with QC-based masks
and writes normalized arrays ready for StaticParameterInverter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pickle

from dataset import DAILY_COLUMN_NAMES, DYNAMIC_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DAILY = BASE_DIR.parent / "OUTPUT_PERTURBATION_LHP_V2_LHS" / "LHP_20250919_daily_0000.txt"
DEFAULT_SCALER = BASE_DIR / "data" / "scaler_dynamic.pkl"
DEFAULT_OBS = BASE_DIR / "observations" / "nep_obs_2012_2016.csv"
DEFAULT_OUTPUT = BASE_DIR / "observations" / "obs_windows_2012_2016.npz"


def load_dynamic_series(daily_path: Path) -> pd.DataFrame:
    if not daily_path.exists():
        raise FileNotFoundError(f"Dynamic daily file not found: {daily_path}")

    df = pd.read_csv(
        daily_path,
        sep=r"\s+",
        header=None,
        names=DAILY_COLUMN_NAMES,
    )
    base = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df["date"] = base + pd.to_timedelta(df["doy"], unit="D")
    df = df.set_index("date")
    return df[DYNAMIC_COLUMNS]


def load_observation_series(obs_csv: Path, start: str, end: str) -> pd.DataFrame:
    if not obs_csv.exists():
        raise FileNotFoundError(f"Observation CSV not found: {obs_csv}")

    obs = pd.read_csv(obs_csv, parse_dates=["date"])
    obs = obs.set_index("date")

    full_index = pd.date_range(start=start, end=end, freq="D")
    obs = obs.reindex(full_index)

    obs_mask = obs["obs_mask"].fillna(False)
    if obs_mask.dtype != bool:
        obs_mask = obs_mask.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    obs["obs_mask"] = obs_mask.astype(bool)
    obs["nep_mg_ha"] = obs["nep_mg_ha"].fillna(0.0)
    return obs


@dataclass
class WindowRecord:
    context_x: np.ndarray
    future_known: np.ndarray
    y_obs: np.ndarray
    obs_mask: np.ndarray
    start_date: pd.Timestamp


def build_windows(
    dynamic_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    scaler,
    context_len: int,
    prediction_len: int,
    min_valid: int,
    start: str,
    end: str,
) -> List[WindowRecord]:
    dynamic_scaled = pd.DataFrame(
        scaler.transform(dynamic_df.values),
        index=dynamic_df.index,
        columns=dynamic_df.columns,
    )

    obs_start = pd.Timestamp(start)
    obs_end = pd.Timestamp(end)
    last_start = obs_end - pd.Timedelta(days=prediction_len - 1)
    window_starts = pd.date_range(obs_start, last_start, freq="D")

    records: List[WindowRecord] = []

    for pred_start in window_starts:
        context_start = pred_start - pd.Timedelta(days=context_len)
        context_end = pred_start - pd.Timedelta(days=1)
        pred_end = pred_start + pd.Timedelta(days=prediction_len - 1)

        ctx_slice = dynamic_scaled.loc[context_start:context_end]
        fut_slice = dynamic_scaled.loc[pred_start:pred_end]
        obs_slice = obs_df.loc[pred_start:pred_end]

        if len(ctx_slice) != context_len:
            continue
        if len(fut_slice) != prediction_len:
            continue
        if len(obs_slice) != prediction_len:
            continue

        mask = obs_slice["obs_mask"].to_numpy(dtype=bool)
        if mask.sum() < min_valid:
            continue

        y_obs = np.zeros((prediction_len, 4), dtype=np.float32)
        y_obs[:, 3] = obs_slice["nep_mg_ha"].to_numpy(dtype=np.float32)

        records.append(
            WindowRecord(
                context_x=ctx_slice.to_numpy(dtype=np.float32),
                future_known=fut_slice.to_numpy(dtype=np.float32),
                y_obs=y_obs,
                obs_mask=mask.astype(bool),
                start_date=pred_start,
            )
        )

    return records


def save_windows(records: List[WindowRecord], output_path: Path) -> None:
    if not records:
        raise RuntimeError("No windows satisfied the filtering conditions")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    context_x = np.stack([r.context_x for r in records])
    future_known = np.stack([r.future_known for r in records])
    y_obs = np.stack([r.y_obs for r in records])
    obs_mask = np.stack([r.obs_mask for r in records])
    window_start = np.array([r.start_date.strftime("%Y-%m-%d") for r in records])

    np.savez_compressed(
        output_path,
        context_x=context_x,
        future_known=future_known,
        y_obs=y_obs,
        obs_mask=obs_mask,
        window_start=window_start,
    )

    print(f"Saved {len(records)} windows to {output_path}")
    print(f"Context shape: {context_x.shape}, Future shape: {future_known.shape}")
    print(f"Valid observations per window (min/median/max): "
          f"{obs_mask.sum(axis=1).min()} / {np.median(obs_mask.sum(axis=1))} / {obs_mask.sum(axis=1).max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build observation windows for inversion")
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY, help="Reference VISIT daily output file")
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER, help="Dynamic scaler path")
    parser.add_argument("--obs", type=Path, default=DEFAULT_OBS, help="Processed observation CSV")
    parser.add_argument("--start", type=str, default="2012-01-01", help="Observation window start date")
    parser.add_argument("--end", type=str, default="2016-12-31", help="Observation window end date")
    parser.add_argument("--context-len", type=int, default=180, help="Context length")
    parser.add_argument("--prediction-len", type=int, default=30, help="Prediction length")
    parser.add_argument("--min-valid", type=int, default=10, help="Minimum valid obs per window")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output NPZ path")

    args = parser.parse_args()

    with open(args.scaler, "rb") as f:
        scaler_dynamic = pickle.load(f)

    dynamic_df = load_dynamic_series(args.daily)
    obs_df = load_observation_series(args.obs, args.start, args.end)

    records = build_windows(
        dynamic_df=dynamic_df,
        obs_df=obs_df,
        scaler=scaler_dynamic,
        context_len=args.context_len,
        prediction_len=args.prediction_len,
        min_valid=args.min_valid,
        start=args.start,
        end=args.end,
    )

    save_windows(records, args.output)


if __name__ == "__main__":
    main()
