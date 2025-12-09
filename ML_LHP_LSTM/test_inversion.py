"""
静的パラメータ逆推定のテスト実行

NEPのみを使って1サンプルの逆推定を行い、結果を可視化する。
観測データ(NEP_obs)にも対応。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime, UTC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import VISITTimeSeriesDataset
from inverse_estimator import StaticParameterInverter, load_inversion_config
from model import create_model

# パス設定
BASE_DIR = Path("/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM")
DATA_DIR = Path("/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OBS_WINDOW_NPZ = BASE_DIR / "observations" / "obs_windows_2012_2016.npz"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "inversion_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static parameter inversion test runner")
    parser.add_argument("--mode", choices=["simulation", "observation"], default="simulation")
    parser.add_argument("--sample-id", type=int, default=0, help="Sample ID for simulation mode")
    parser.add_argument("--window-index", type=int, default=100, help="Window index for simulation mode")
    parser.add_argument("--obs-index", type=int, default=0, help="Observation window index")
    parser.add_argument("--obs-range-start", type=str, default=None, help="Start date for observation window range (inclusive)")
    parser.add_argument("--obs-range-end", type=str, default=None, help="End date for observation window range (inclusive)")
    parser.add_argument("--obs-range-step", type=int, default=None, help="Subsample every Nth window within range")
    parser.add_argument("--max-obs-windows", type=int, default=None, help="Limit number of observation windows when using a range")
    parser.add_argument("--obs-prior-sample", type=int, default=None, help="Sample ID to use as prior in observation mode")
    parser.add_argument("--obs-npz", type=Path, default=OBS_WINDOW_NPZ, help="Observation NPZ path")
    parser.add_argument("--result-dir", type=Path, default=BASE_DIR / "inverse_results", help="Directory to store outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Inversion config YAML (optional)")
    parser.add_argument("--unconstrained", action="store_true", help="Disable clipping and prior regularization")
    parser.add_argument("--param-save", type=Path, default=None, help="Path to save parameter snapshot (JSON)")
    parser.add_argument("--param-load", type=Path, default=None, help="Load optimized parameters from snapshot and skip inversion")
    return parser.parse_args()


def load_model(device: str):
    with open(ARTIFACTS_DIR / "config.json", "r") as f:
        config = json.load(f)

    model = create_model(
        dynamic_dim=config["dynamic_dim"],
        static_dim=config["static_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
        device=device,
    )

    checkpoint = torch.load(
        ARTIFACTS_DIR / "checkpoint_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint["epoch"]


def load_param_table():
    param_df = pd.read_csv(DATA_DIR / "parameter_summary.csv")
    sample_ids = param_df["sample_id"].to_numpy()
    value_cols = [col for col in param_df.columns if col != "sample_id"]
    values = param_df[value_cols].to_numpy()
    lookup = {sid: values[idx] for idx, sid in enumerate(sample_ids)}
    params_mean = values.mean(axis=0)
    return lookup, params_mean, value_cols


def ensure_batch(arr: np.ndarray, base_ndim: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == base_ndim:
        return arr[None, ...]
    return arr


def build_window_dates(window_starts: pd.Series, horizon: int) -> np.ndarray:
    offsets = np.arange(horizon).astype("timedelta64[D]")
    dates = np.empty((len(window_starts), horizon), dtype="datetime64[ns]")
    for idx, start in enumerate(window_starts):
        base = np.datetime64(start.to_datetime64())
        dates[idx] = base + offsets
    return dates


def save_param_snapshot(
    path: Path,
    param_names: list[str],
    params_prior: np.ndarray,
    params_opt: np.ndarray,
    metadata: dict | None = None,
) -> None:
    payload = {
        "param_names": param_names,
        "prior": params_prior.tolist(),
        "optimized": params_opt.tolist(),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_param_snapshot(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if "optimized" not in data:
        raise ValueError("Parameter snapshot missing 'optimized' field")
    return data


def write_param_table(path: Path, param_names: list[str], prior: np.ndarray, optimized: np.ndarray) -> None:
    delta = optimized - prior
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_pct = np.where(np.abs(prior) > 1e-8, 100.0 * delta / prior, np.nan)
    df = pd.DataFrame(
        {
            "param_name": param_names,
            "prior": prior,
            "optimized": optimized,
            "delta": delta,
            "delta_pct": delta_pct,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def aggregate_daily_nep(
    window_dates: np.ndarray,
    y_obs: np.ndarray,
    pred_prior: np.ndarray,
    pred_opt: np.ndarray,
    obs_mask: np.ndarray,
) -> pd.DataFrame:
    flat_dates = pd.to_datetime(window_dates.reshape(-1))
    flat_mask = obs_mask.reshape(-1).astype(bool)
    flat_obs = y_obs[:, :, 3].reshape(-1)
    flat_prior = pred_prior[:, :, 3].reshape(-1)
    flat_opt = pred_opt[:, :, 3].reshape(-1)

    df = pd.DataFrame(
        {
            "date": flat_dates,
            "mask": flat_mask,
            "obs": flat_obs,
            "prior": flat_prior,
            "opt": flat_opt,
        }
    )
    df = df[df["mask"]]
    if df.empty:
        return pd.DataFrame(columns=["date", "obs", "prior", "opt", "count"]).astype({"date": "datetime64[ns]"})

    grouped = (
        df.groupby("date")
        .agg(obs=("obs", "first"), prior=("prior", "mean"), opt=("opt", "mean"), count=("obs", "size"))
        .reset_index()
        .sort_values("date")
    )
    return grouped


def select_observation_indices(window_start: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.obs_range_start is None:
        return np.array([args.obs_index], dtype=int)

    starts = pd.to_datetime(window_start)
    range_start = pd.Timestamp(args.obs_range_start)
    range_end = pd.Timestamp(args.obs_range_end or args.obs_range_start)
    mask = (starts >= range_start) & (starts <= range_end)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise ValueError("No observation windows fall within the specified range")

    step = args.obs_range_step or 1
    idxs = idxs[::step]
    if args.max_obs_windows is not None:
        idxs = idxs[: args.max_obs_windows]
    return idxs


def compute_masked_rmse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    diff = pred[mask] - obs[mask]
    return float(np.sqrt(np.mean(diff**2)))


def main() -> None:
    args = parse_args()
    if args.param_load and not args.param_load.exists():
        raise FileNotFoundError(f"param-load path not found: {args.param_load}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("Static Parameter Inversion Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    model, config, epoch = load_model(device)

    with open(BASE_DIR / "data/scaler_dynamic.pkl", "rb") as f:
        scaler_dynamic = pickle.load(f)
    with open(BASE_DIR / "data/scaler_static.pkl", "rb") as f:
        scaler_static = pickle.load(f)

    print("[Step 1] Model + scaler ready")
    print(f"  ✓ checkpoint epoch {epoch}")

    param_lookup, params_mean, param_names = load_param_table()
    window_dates = None

    if args.mode == "simulation":
        print("\n[Step 2] Preparing VISIT window (test split)")
        test_dataset = VISITTimeSeriesDataset(
            data_dir=str(DATA_DIR),
            split="test",
            context_len=config["context_len"],
            prediction_len=config["prediction_len"],
            scaler_dynamic=scaler_dynamic,
            scaler_static=scaler_static,
            fit_scaler=False,
        )

        if not 0 <= args.window_index < len(test_dataset):
            raise IndexError(f"window-index out of range (0-{len(test_dataset)-1})")

        test_window = test_dataset[args.window_index]
        context_x = test_window["context_x"].numpy()
        future_known = test_window["future_known"].numpy()
        y_obs = test_window["target_y"].numpy()
        obs_mask = np.ones(len(y_obs), dtype=bool)
        params_prior_raw = param_lookup.get(args.sample_id)
        if params_prior_raw is None:
            raise KeyError(f"sample-id {args.sample_id} not found in parameter_summary.csv")
        window_label = f"test_window_{args.window_index}"
        print(f"  ✓ Selected sample {args.sample_id}, window {args.window_index}")
    else:
        print("\n[Step 2] Preparing observation window(s)")
        data = np.load(args.obs_npz)
        total_windows = data["context_x"].shape[0]
        selected_indices = select_observation_indices(data["window_start"], args)
        for idx in selected_indices:
            if not 0 <= idx < total_windows:
                raise IndexError(f"obs index {idx} out of range (0-{total_windows-1})")

        context_x = data["context_x"][selected_indices]
        future_known = data["future_known"][selected_indices]
        y_obs = data["y_obs"][selected_indices]
        obs_mask = data["obs_mask"][selected_indices].astype(bool)
        window_starts = pd.to_datetime(data["window_start"][selected_indices])
        window_dates = build_window_dates(window_starts, future_known.shape[1])
        window_label = (
            f"obs_{window_starts[0].date()}_to_{window_starts[-1].date()}"
            if len(window_starts) > 1
            else f"obs_{window_starts[0].date()}"
        )

        if args.obs_prior_sample is not None:
            params_prior_raw = param_lookup.get(args.obs_prior_sample)
            if params_prior_raw is None:
                raise KeyError(f"sample-id {args.obs_prior_sample} not found for prior")
        else:
            params_prior_raw = params_mean.copy()

        total_valid = int(obs_mask.sum())
        total_points = int(obs_mask.size)
        print(
            f"  ✓ Windows: {len(selected_indices)} (first={window_starts[0].date()}, last={window_starts[-1].date()})"
        )
        print(f"  ✓ Valid obs: {total_valid} / {total_points}")

    context_x = ensure_batch(context_x, 2)
    future_known = ensure_batch(future_known, 2)
    y_obs = ensure_batch(y_obs, 2)
    obs_mask = ensure_batch(obs_mask, 1)

    dynamic_dim = context_x.shape[-1]
    future_dim = future_known.shape[-1]
    print(f"  ✓ Context shape: {context_x.shape} (dynamic_dim={dynamic_dim})")
    print(f"  ✓ Future known shape: {future_known.shape} (future_dim={future_dim})")

    config_path = args.config if args.config and args.config.exists() else None
    inversion_config = load_inversion_config(config_path)
    if config_path:
        print(f"  ✓ Loaded inversion config: {config_path}")
    else:
        print("  ✓ Using default inversion config")
    if args.unconstrained:
        print("  ⚠ Unconstrained mode enabled (no clipping / prior regularization)")
    params_opt_raw = None
    info = {}
    if args.param_load is None:
        inverter = StaticParameterInverter(
            model=model,
            static_scaler=scaler_static,
            dynamic_scaler=scaler_dynamic,
            config=inversion_config,
            device=device,
            param_names=param_names,
        )

        params_opt_raw, info = inverter.invert(
            params_prior_raw=params_prior_raw,
            dynamic_input=context_x,
            future_known=future_known,
            y_obs=y_obs,
            obs_mask=obs_mask,
            unconstrained=args.unconstrained,
        )
    else:
        snapshot = load_param_snapshot(args.param_load)
        params_opt_raw = np.asarray(snapshot["optimized"], dtype=float)
        snapshot_prior = snapshot.get("prior")
        if snapshot_prior is not None:
            params_prior_raw = np.asarray(snapshot_prior, dtype=float)
        snapshot_names = snapshot.get("param_names")
        if snapshot_names and list(snapshot_names) != list(param_names):
            print("  ⚠ Warning: param names in snapshot differ from current table")
        info = {
            "loss_history": snapshot.get("metadata", {}).get("loss_history", []),
            "final_loss": snapshot.get("metadata", {}).get("final_loss", float("nan")),
            "iterations": snapshot.get("metadata", {}).get("iterations", 0),
        }

    param_change_abs = np.abs(params_opt_raw - params_prior_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        param_change_pct = np.where(
            np.abs(params_prior_raw) > 1e-8,
            100.0 * param_change_abs / np.abs(params_prior_raw),
            np.nan,
        )
    info.setdefault("param_change_abs", param_change_abs)
    info.setdefault("param_change_pct", param_change_pct)

    print("\n[Step 3] Evaluating predictions")
    params_prior_norm = scaler_static.transform(params_prior_raw.reshape(1, -1))[0]
    context_x_t = torch.tensor(context_x, dtype=torch.float32, device=device)
    future_known_t = torch.tensor(future_known, dtype=torch.float32, device=device)
    batch_size = context_x_t.shape[0]

    with torch.no_grad():
        params_prior_t = torch.tensor(params_prior_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        pred_prior = model(context_x_t, params_prior_t, future_known_t).cpu().numpy()

    params_opt_norm = scaler_static.transform(params_opt_raw.reshape(1, -1))[0]
    with torch.no_grad():
        params_opt_t = torch.tensor(params_opt_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
        pred_opt = model(context_x_t, params_opt_t, future_known_t).cpu().numpy()

    if args.mode == "observation" and window_dates is not None:
        daily_series = aggregate_daily_nep(window_dates, y_obs, pred_prior, pred_opt, obs_mask)
        if daily_series.empty:
            rmse_prior = float("nan")
            rmse_opt = float("nan")
            plot_x = np.array([])
            obs_nep_plot = np.array([])
            pred_prior_plot = np.array([])
            pred_opt_plot = np.array([])
            valid_obs_summary = 0
        else:
            daily_mask = np.ones(len(daily_series), dtype=bool)
            rmse_prior = compute_masked_rmse(
                daily_series["prior"].to_numpy(), daily_series["obs"].to_numpy(), daily_mask
            )
            rmse_opt = compute_masked_rmse(
                daily_series["opt"].to_numpy(), daily_series["obs"].to_numpy(), daily_mask
            )
            plot_x = pd.to_datetime(daily_series["date"]).to_numpy()
            obs_nep_plot = daily_series["obs"].to_numpy()
            pred_prior_plot = daily_series["prior"].to_numpy()
            pred_opt_plot = daily_series["opt"].to_numpy()
            valid_obs_summary = len(daily_series)
        total_obs_summary = len(np.unique(window_dates.reshape(-1)))
    else:
        rmse_prior = compute_masked_rmse(pred_prior[:, :, 3], y_obs[:, :, 3], obs_mask)
        rmse_opt = compute_masked_rmse(pred_opt[:, :, 3], y_obs[:, :, 3], obs_mask)
        y_obs_flat = y_obs.reshape(-1, y_obs.shape[-1])
        pred_prior_flat = pred_prior.reshape(-1, pred_prior.shape[-1])
        pred_opt_flat = pred_opt.reshape(-1, pred_opt.shape[-1])
        plot_x = np.arange(y_obs_flat.shape[0])
        obs_nep_plot = y_obs_flat[:, 3]
        pred_prior_plot = pred_prior_flat[:, 3]
        pred_opt_plot = pred_opt_flat[:, 3]
        obs_mask_flat = obs_mask.reshape(-1)
        valid_obs_summary = int(obs_mask_flat.sum())
        total_obs_summary = int(obs_mask_flat.size)
    print("  ✓ Masked NEP RMSE")
    print(f"      Prior:     {rmse_prior:.6f}")
    print(f"      Optimized: {rmse_opt:.6f}")
    if np.isfinite(rmse_prior):
        improvement = 100 * (rmse_prior - rmse_opt) / rmse_prior
        print(f"      Improvement: {improvement:.2f}%")
    else:
        improvement = float("nan")

    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    param_table_path = result_dir / "parameter_values.csv"
    write_param_table(param_table_path, param_names, params_prior_raw, params_opt_raw)

    param_snapshot_path = args.param_save if args.param_save else (result_dir / "params_snapshot.json")
    snapshot_metadata = {
        "mode": args.mode,
        "window_label": window_label,
        "obs_range_start": args.obs_range_start,
        "obs_range_end": args.obs_range_end,
        "obs_index": args.obs_index,
        "config": str(config_path) if config_path else None,
        "unconstrained": args.unconstrained,
        "param_load": str(args.param_load) if args.param_load else None,
        "loss_history": info.get("loss_history", []),
        "final_loss": info.get("final_loss"),
        "iterations": info.get("iterations"),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    save_param_snapshot(param_snapshot_path, param_names, params_prior_raw, params_opt_raw, snapshot_metadata)

    # 図1: NEP比較
    print("\n[Step 4] Saving visualizations")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax = axes[0]
    ax.plot(plot_x, obs_nep_plot, "g-", linewidth=2.5, label="Observation", alpha=0.8)
    ax.plot(plot_x, pred_prior_plot, "b--", linewidth=2, label=f"Prior (RMSE={rmse_prior:.4f})", alpha=0.7)
    ax.plot(plot_x, pred_opt_plot, "r--", linewidth=2, label=f"Optimized (RMSE={rmse_opt:.4f})", alpha=0.7)
    x_label = "Date" if args.mode == "observation" and window_dates is not None else "Days ahead"
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("NEP (MgC/ha/day)", fontsize=12, fontweight="bold")
    ax.set_title("NEP Prediction: Before vs After", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    error_prior = pred_prior_plot - obs_nep_plot
    error_opt = pred_opt_plot - obs_nep_plot
    ax.plot(plot_x, error_prior, "b-", linewidth=2, label="Prior error", alpha=0.7)
    ax.plot(plot_x, error_opt, "r-", linewidth=2, label="Optimized error", alpha=0.7)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.fill_between(plot_x, 0, error_prior, alpha=0.2, color="blue")
    ax.fill_between(plot_x, 0, error_opt, alpha=0.2, color="red")
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Prediction Error (MgC/ha/day)", fontsize=12, fontweight="bold")
    ax.set_title("Prediction Errors", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_dir / "nep_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ nep_comparison.png")

    loss_history = info.get("loss_history", [])
    if loss_history:
        fig, ax = plt.subplots(figsize=(12, 6))
        iterations = range(len(loss_history))
        total_losses = [h.get("total", float("nan")) for h in loss_history]
        obs_losses = [h.get("obs", float("nan")) for h in loss_history]
        prior_losses = [h.get("prior", float("nan")) for h in loss_history]
        ax.plot(iterations, total_losses, "k-", linewidth=2.5, label="Total Loss", alpha=0.8)
        ax.plot(iterations, obs_losses, "g-", linewidth=2, label="Observation Loss", alpha=0.7)
        ax.plot(iterations, prior_losses, "b-", linewidth=2, label="Prior Loss", alpha=0.7)
        ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title("Loss History", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(result_dir / "loss_history.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ loss_history.png")
    else:
        print("  - Skipping loss history plot (no optimization trace)")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    param_indices = np.arange(len(params_prior_raw))
    ax = axes[0]
    ax.bar(param_indices, param_change_abs, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Parameter Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Absolute Change", fontsize=12, fontweight="bold")
    ax.set_title("Absolute Parameter Changes", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(param_indices, param_change_pct, alpha=0.7, edgecolor="black", color="orange")
    ax.set_xlabel("Parameter Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Change (%)", fontsize=12, fontweight="bold")
    ax.set_title("Relative Parameter Changes", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(result_dir / "parameter_changes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ parameter_changes.png")

    print("\n[Step 5] Writing summary")
    summary_path = result_dir / "inversion_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Parameter Inversion Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Window label: {window_label}\n")
        f.write(f"Prediction Length: {config['prediction_len']} days\n")
        f.write(f"Valid observations: {valid_obs_summary} / {total_obs_summary}\n\n")
        f.write("--- NEP RMSE ---\n")
        f.write(f"Prior:     {rmse_prior:.6f}\n")
        f.write(f"Optimized: {rmse_opt:.6f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n\n")
        f.write("--- Parameter Changes ---\n")
        f.write(f"Mean absolute change: {np.nanmean(param_change_abs):.6f}\n")
        f.write(f"Max absolute change:  {np.nanmax(param_change_abs):.6f}\n")
        f.write(f"Mean relative change: {np.nanmean(param_change_pct):.2f}%\n")
        f.write(f"Max relative change:  {np.nanmax(param_change_pct):.2f}%\n\n")
        f.write("--- Optimization ---\n")
        f.write(f"Iterations: {info.get('iterations', 0)}\n")
        final_loss = info.get("final_loss")
        if final_loss is not None and np.isfinite(final_loss):
            f.write(f"Final loss: {final_loss:.6f}\n")
        else:
            f.write("Final loss: N/A (loaded parameters)\n")
        f.write("=" * 70 + "\n")
    print(f"  ✓ {summary_path.name}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print(f"Results saved in: {result_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
