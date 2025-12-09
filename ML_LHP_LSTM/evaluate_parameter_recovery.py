"""NEPシナリオの静的パラメータ復元精度を評価するスクリプト。

`test_inversion.py` と同じ LSTM サロゲートと逆推定器を用い、
指定したサンプル/ウィンドウに対して NEP からパラメータを推定し、
実際に使用されたパラメータとの一致度をレポートする。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from datetime import datetime, UTC
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for batch plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import VISITTimeSeriesDataset
from inverse_estimator import StaticParameterInverter, load_inversion_config
from model import create_model

BASE_DIR = Path("/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM")
DATA_DIR = Path("/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "inversion_config.yaml"
DEFAULT_RESULT_DIR = BASE_DIR / "inverse_recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate parameter recovery via surrogate inversion")
    parser.add_argument("--window-index", type=int, default=0, help="test split window index to invert")
    parser.add_argument("--sample-id", type=int, default=None, help="override true sample_id (defaults to window metadata)")
    parser.add_argument(
        "--prior-mode",
        choices=["true", "mean", "sample"],
        default="true",
        help="how to choose prior parameters before inversion",
    )
    parser.add_argument(
        "--prior-sample-id",
        type=int,
        default=None,
        help="sample_id to use when prior-mode=sample",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="inversion config YAML")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR, help="directory to store outputs")
    parser.add_argument("--unconstrained", action="store_true", help="disable clipping + prior regularization")
    parser.add_argument("--no-plots", dest="plot", action="store_false", help="skip generating result figures")
    parser.add_argument(
        "--trace-params",
        nargs="+",
        default=None,
        help="parameter names to record per-iteration and plot (e.g., tree_topt tree_pmax)",
    )
    parser.add_argument(
        "--trace-stride",
        type=int,
        default=1,
        help="subsample factor when plotting parameter traces (default=1 meaning every iteration)",
    )
    parser.set_defaults(plot=True)
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

    checkpoint = torch.load(ARTIFACTS_DIR / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint.get("epoch", -1)


def load_scalers():
    with open(BASE_DIR / "data" / "scaler_dynamic.pkl", "rb") as f:
        scaler_dynamic = pickle.load(f)
    with open(BASE_DIR / "data" / "scaler_static.pkl", "rb") as f:
        scaler_static = pickle.load(f)
    return scaler_dynamic, scaler_static


def load_parameter_table() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_DIR / "parameter_summary.csv")
    param_cols = [c for c in df.columns if c != "sample_id"]
    return df.set_index("sample_id"), param_cols


def ensure_batch(arr: np.ndarray, base_ndim: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == base_ndim:
        return arr[None, ...]
    return arr


def build_prior(
    prior_mode: str,
    prior_sample_id: int | None,
    param_table: pd.DataFrame,
    param_cols: list[str],
    true_sample_id: int,
) -> np.ndarray:
    if prior_mode == "true":
        return param_table.loc[true_sample_id, param_cols].to_numpy()
    if prior_mode == "mean":
        return param_table[param_cols].to_numpy().mean(axis=0)
    if prior_mode == "sample":
        if prior_sample_id is None:
            raise ValueError("prior-mode=sample の場合は --prior-sample-id を指定してください")
        if prior_sample_id not in param_table.index:
            raise KeyError(f"prior sample_id {prior_sample_id} が parameter_summary.csv に存在しません")
        return param_table.loc[prior_sample_id, param_cols].to_numpy()
    raise ValueError(f"Unknown prior_mode: {prior_mode}")


def predict_with_params(
    model: torch.nn.Module,
    scaler_static,
    context_x: np.ndarray,
    future_known: np.ndarray,
    params_raw: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run the surrogate model with specified static parameters and return predictions."""

    params_norm = scaler_static.transform(params_raw.reshape(1, -1))[0]
    context_t = torch.tensor(context_x, dtype=torch.float32, device=device)
    future_t = torch.tensor(future_known, dtype=torch.float32, device=device)
    params_t = torch.tensor(params_norm, dtype=torch.float32, device=device)
    batch = context_t.shape[0]
    with torch.no_grad():
        preds = model(context_t, params_t.unsqueeze(0).repeat(batch, 1), future_t)
    return preds.cpu().numpy()


def plot_param_traces(
    param_cols: list[str],
    history: np.ndarray,
    params_true: np.ndarray,
    trace_names: list[str],
    stride: int,
    result_dir: Path,
    stem: str,
) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(param_cols)}
    selected = []
    for name in trace_names:
        idx = name_to_idx.get(name)
        if idx is None:
            print(f"⚠ trace parameter '{name}' は存在しません。スキップします。")
            continue
        selected.append((name, idx))

    if not selected:
        print("⚠ 有効な trace パラメータが指定されなかったため、軌跡図は作成しません。")
        return

    stride = max(1, stride)
    history_sub = history[::stride]
    iterations = np.arange(history_sub.shape[0]) * stride

    n = len(selected)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for ax, (name, idx) in zip(axes.flat, selected):
        ax.plot(iterations, history_sub[:, idx], label="Estimated", color="#1f618d")
        ax.axhline(params_true[idx], color="#c0392b", linestyle="--", label="True")
        final_val = history_sub[-1, idx]
        abs_err = abs(final_val - params_true[idx])
        rel_err = 100.0 * abs_err / (abs(params_true[idx]) + 1e-8)
        ax.set_title(f"{name}\nError={abs_err:.4f} ({rel_err:.1f}%)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    # Hide unused axes
    for ax in axes.flat[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = result_dir / f"{stem}_param_traces.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_plots(
    param_cols: list[str],
    params_true: np.ndarray,
    params_prior: np.ndarray,
    params_est: np.ndarray,
    abs_err: np.ndarray,
    y_obs: np.ndarray,
    pred_true: np.ndarray,
    pred_prior: np.ndarray,
    pred_est: np.ndarray,
    result_dir: Path,
    stem: str,
) -> None:
    """Generate overview plots comparing parameters and NEP trajectories."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)

    # Scatter: true vs estimated parameters
    ax = axes[0]
    ax.scatter(params_true, params_est, c="#2874a6", alpha=0.8, label="Estimated")
    min_val = float(min(params_true.min(), params_est.min()))
    max_val = float(max(params_true.max(), params_est.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", label="y=x")
    ax.set_title("Parameter recovery (true vs estimated)")
    ax.set_xlabel("True parameter value")
    ax.set_ylabel("Estimated parameter value")
    ax.legend()

    # Horizontal bar: absolute error per parameter (sorted)
    ax = axes[1]
    sort_idx = np.argsort(abs_err)
    ax.barh(
        np.array(param_cols)[sort_idx],
        abs_err[sort_idx],
        color="#f39c12",
        alpha=0.8,
    )
    ax.set_title("Absolute error per parameter (sorted)")
    ax.set_xlabel("|estimated - true|")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # NEP comparison over prediction horizon
    ax = axes[2]
    days = np.arange(y_obs.shape[1])
    nep_obs = y_obs[0, :, 3]
    nep_true = pred_true[0, :, 3]
    nep_prior = pred_prior[0, :, 3]
    nep_est = pred_est[0, :, 3]

    ax.plot(days, nep_obs, label="VISIT target (NEP)", color="#212f3d", linewidth=2)
    ax.plot(days, nep_true, label="Surrogate (true params)", linestyle="--", color="#117a65")
    ax.plot(days, nep_prior, label="Surrogate (prior params)", linestyle=":", color="#b03a2e")
    ax.plot(days, nep_est, label="Surrogate (estimated params)", linestyle="-", color="#1f618d")
    ax.set_title("NEP trajectory comparison")
    ax.set_xlabel("Prediction step (day)")
    ax.set_ylabel("NEP")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    out_path = result_dir / f"{stem}_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Optional detailed per-parameter comparison (true/prior/estimated)
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    x = np.arange(len(param_cols))
    width = 0.28
    ax2.bar(x - width, params_true, width=width, label="True", color="#148f77")
    ax2.bar(x, params_prior, width=width, label="Prior", color="#cd6155")
    ax2.bar(x + width, params_est, width=width, label="Estimated", color="#5dade2")
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_cols, rotation=90)
    ax2.set_ylabel("Parameter value")
    ax2.set_title("Parameter values (true / prior / estimated)")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    out_path2 = result_dir / f"{stem}_param_bars.png"
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Parameter Recovery Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Window index: {args.window_index}")
    print(f"Prior mode: {args.prior_mode}")
    if args.prior_mode == "sample":
        print(f"Prior sample_id: {args.prior_sample_id}")
    print(f"Unconstrained: {args.unconstrained}")
    print("=" * 80)

    model, config, epoch = load_model(device)
    scaler_dynamic, scaler_static = load_scalers()
    print(f"✓ Model loaded (epoch {epoch})")
    print("✓ Scalers loaded")

    param_table, param_cols = load_parameter_table()

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
        raise IndexError(f"window-index {args.window_index} は 0〜{len(test_dataset)-1} の範囲で指定してください")

    window = test_dataset[args.window_index]
    window_meta = test_dataset.windows[args.window_index]
    dataset_sample_idx = window_meta["sample_id"]
    sample_id_from_window = int(test_dataset.sample_ids[dataset_sample_idx])
    if args.sample_id is not None and args.sample_id != sample_id_from_window:
        print(
            f"⚠ 指定 sample_id({args.sample_id}) とウィンドウ由来 ({sample_id_from_window}) が異なるため、"
            "実際のサンプルIDとして前者を優先します"
        )
        true_sample_id = args.sample_id
    else:
        true_sample_id = sample_id_from_window

    if true_sample_id not in param_table.index:
        raise KeyError(f"sample_id {true_sample_id} が parameter_summary.csv に存在しません")

    params_true_raw = param_table.loc[true_sample_id, param_cols].to_numpy()
    params_prior_raw = build_prior(args.prior_mode, args.prior_sample_id, param_table, param_cols, true_sample_id)

    context_x = ensure_batch(window["context_x"].numpy(), 2)
    future_known = ensure_batch(window["future_known"].numpy(), 2)
    y_obs = ensure_batch(window["target_y"].numpy(), 2)
    obs_mask = np.ones(y_obs.shape[:-1], dtype=bool)

    inversion_config = load_inversion_config(args.config if args.config and args.config.exists() else None)
    inverter = StaticParameterInverter(
        model=model,
        static_scaler=scaler_static,
        dynamic_scaler=scaler_dynamic,
        config=inversion_config,
        device=device,
        param_names=param_cols,
    )

    params_opt_raw, info = inverter.invert(
        params_prior_raw=params_prior_raw,
        dynamic_input=context_x,
        future_known=future_known,
        y_obs=y_obs,
        obs_mask=obs_mask,
        unconstrained=args.unconstrained,
        record_history=bool(args.trace_params),
    )

    diff = params_opt_raw - params_true_raw
    abs_err = np.abs(diff)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_pct = np.where(np.abs(params_true_raw) > 1e-8, 100.0 * diff / params_true_raw, np.nan)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(abs_err))

    pred_true = predict_with_params(model, scaler_static, context_x, future_known, params_true_raw, device)
    pred_prior = predict_with_params(model, scaler_static, context_x, future_known, params_prior_raw, device)
    pred_est = predict_with_params(model, scaler_static, context_x, future_known, params_opt_raw, device)

    print("\n結果サマリ (推定 vs 真値)")
    print("-" * 40)
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAX:  {max_abs:.6f}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = f"sample{true_sample_id:03d}_win{args.window_index:04d}_{args.prior_mode}_{timestamp}"
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        {
            "param_name": param_cols,
            "true": params_true_raw,
            "prior": params_prior_raw,
            "estimated": params_opt_raw,
            "abs_err": abs_err,
            "rel_err_pct": rel_err_pct,
        }
    )
    table_path = result_dir / f"{stem}_params.csv"
    table.to_csv(table_path, index=False)

    summary = {
        "sample_id": true_sample_id,
        "window_index": args.window_index,
        "dataset_sample_idx": dataset_sample_idx,
        "prior_mode": args.prior_mode,
        "prior_sample_id": args.prior_sample_id,
        "unconstrained": args.unconstrained,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "final_loss": info.get("final_loss"),
        "iterations": info.get("iterations"),
    }
    summary_path = result_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.plot:
        create_plots(
            param_cols=param_cols,
            params_true=params_true_raw,
            params_prior=params_prior_raw,
            params_est=params_opt_raw,
            abs_err=abs_err,
            y_obs=y_obs,
            pred_true=pred_true,
            pred_prior=pred_prior,
            pred_est=pred_est,
            result_dir=result_dir,
            stem=stem,
        )
        print("✓ グラフを出力しました (overview / param_bars)")

        if args.trace_params and "param_history_raw" in info:
            plot_param_traces(
                param_cols=param_cols,
                history=info["param_history_raw"],
                params_true=params_true_raw,
                trace_names=args.trace_params,
                stride=args.trace_stride,
                result_dir=result_dir,
                stem=stem,
            )
            print("✓ パラメータ推移グラフを出力しました (param_traces)")

    print(f"\n✓ 結果を書き出しました: {table_path}")
    print(f"✓ サマリ: {summary_path}")


if __name__ == "__main__":
    main()
