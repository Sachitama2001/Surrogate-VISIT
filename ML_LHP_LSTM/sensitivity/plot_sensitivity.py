"""Plotting and result-saving utilities for Sobol sensitivity analysis."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _ensure_dirs(result_dir: Path, figure_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)


def _save_arrays(result_dir: Path, qoi_name: str, problem: dict, param_values: np.ndarray, y: np.ndarray, sobol_result: dict) -> None:
    np.save(result_dir / f"qoi_values_{qoi_name}.npy", y)
    np.savez(result_dir / f"param_samples_{qoi_name}.npz", param_values=param_values, param_names=problem["names"])

    df = pd.DataFrame(
        {
            "param": problem["names"],
            "S1": sobol_result.get("S1"),
            "S1_conf": sobol_result.get("S1_conf"),
            "ST": sobol_result.get("ST"),
            "ST_conf": sobol_result.get("ST_conf"),
        }
    )
    df.to_csv(result_dir / f"sobol_indices_{qoi_name}.csv", index=False)

    json_data = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sobol_result.items()}
    json_data["params"] = problem["names"]
    (result_dir / f"sobol_indices_{qoi_name}.json").write_text(json.dumps(json_data, indent=2))


def _plot_bar(values: np.ndarray, conf: np.ndarray, names: List[str], title: str, out_path: Path, top_k: int = 15) -> None:
    order = np.argsort(np.abs(values))[::-1]
    order = order[: min(top_k, len(order))]
    vals = values[order]
    errs = conf[order]
    labels = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(vals)), vals, yerr=errs, capsize=3, color="steelblue")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(title.split()[0])
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scatter_s1_st(s1: np.ndarray, st: np.ndarray, names: List[str], out_path: Path, top_k: int = 15) -> None:
    order = np.argsort(np.abs(s1))[::-1]
    order = order[: min(top_k, len(order))]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s1[order], st[order], color="darkorange", edgecolor="k", s=60)
    for idx in order:
        ax.annotate(names[idx], (s1[idx], st[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("S1")
    ax.set_ylabel("ST")
    ax.set_title("S1 vs ST (top parameters)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_s2_heatmap(s2: np.ndarray, names: List[str], out_path: Path, top_k: int = 10) -> None:
    if s2 is None:
        return
    order = np.argsort(np.abs(np.nan_to_num(s2, nan=0.0)).sum(axis=0))[::-1]
    order = order[: min(top_k, len(order))]
    s2_sub = np.abs(s2[np.ix_(order, order)])
    np.fill_diagonal(s2_sub, np.nan)
    labels = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(s2_sub, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|S2|")
    ax.set_title("Second-order Sobol |S2| (top interactions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _predict_qoi(static_scaled: np.ndarray, model, device: str, context_x: torch.Tensor, future_known: torch.Tensor, qoi_fn, batch_size: int) -> np.ndarray:
    n = static_scaled.shape[0]
    results = np.zeros(n, dtype=np.float64)
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            static_batch = torch.from_numpy(static_scaled[start:end]).float().to(device)
            context_batch = context_x.repeat(end - start, 1, 1)
            future_batch = future_known.repeat(end - start, 1, 1)
            pred = model(context_batch, static_batch, future_batch)
            qoi_val = qoi_fn(pred)
            results[start:end] = qoi_val.cpu().numpy()
    return results


def _plot_pdp(
    top_params: List[str],
    bounds: dict,
    baseline: np.ndarray,
    scaler,
    model,
    device: str,
    context_x: torch.Tensor,
    future_known: torch.Tensor,
    qoi_fn,
    out_dir: Path,
    batch_size: int,
    param_names: List[str],
) -> None:
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)

    for param in top_params:
        if param not in bounds:
            continue
        param_min = bounds[param]["min"]
        param_max = bounds[param]["max"]
        grid = np.linspace(param_min, param_max, 40)

        static_matrix = np.repeat(baseline[None, :], len(grid), axis=0)
        if param not in param_names:
            continue
        idx = param_names.index(param)
        static_matrix[:, idx] = grid

        static_scaled = (static_matrix - mean) / scale
        y = _predict_qoi(static_scaled, model, device, context_x, future_known, qoi_fn, batch_size)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, y, color="teal", lw=2)
        ax.set_xlabel(f"{param} (physical units)")
        ax.set_ylabel("QoI")
        ax.set_title(f"Partial Dependence: {param}")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{param}_partial_dependence.png", dpi=150)
        plt.close(fig)


def save_results_and_figures(
    problem: dict,
    param_values: np.ndarray,
    y: np.ndarray,
    sobol_result: dict,
    config: dict,
    param_names: List[str],
    static_baseline_raw: np.ndarray,
    static_scaler,
    qoi_fn,
    model,
    device: str,
    context_x: torch.Tensor,
    future_known: torch.Tensor,
):
    qoi_name = config.get("qoi", {}).get("name", "qoi")
    out_cfg = config.get("output", {})
    result_dir = Path(out_cfg.get("base_dir", "sensitivity/results")) / out_cfg.get("tag", "run")
    figure_dir = Path(out_cfg.get("figures_dir", "sensitivity/figures")) / out_cfg.get("tag", "run")
    _ensure_dirs(result_dir, figure_dir)

    _save_arrays(result_dir, qoi_name, problem, param_values, y, sobol_result)

    s1 = np.asarray(sobol_result.get("S1"))
    st = np.asarray(sobol_result.get("ST"))
    s1_conf = np.asarray(sobol_result.get("S1_conf"))
    st_conf = np.asarray(sobol_result.get("ST_conf"))
    s2 = sobol_result.get("S2")

    _plot_bar(s1, s1_conf, problem["names"], "S1 (first-order)", figure_dir / f"{qoi_name}_S1_bar.png")
    _plot_bar(st, st_conf, problem["names"], "ST (total-order)", figure_dir / f"{qoi_name}_ST_bar.png")
    _plot_scatter_s1_st(s1, st, problem["names"], figure_dir / f"{qoi_name}_S1_vs_ST_scatter.png")
    if s2 is not None:
        _plot_s2_heatmap(np.asarray(s2), problem["names"], figure_dir / f"{qoi_name}_S2_heatmap_topK.png")

    # Partial dependence for top-3 by |S1|
    order = np.argsort(np.abs(s1))[::-1]
    top_params = [problem["names"][i] for i in order[: min(3, len(order))]]
    bounds = {name: {"min": b[0], "max": b[1]} for name, b in zip(problem["names"], problem["bounds"])}
    _plot_pdp(
        top_params=top_params,
        bounds=bounds,
        baseline=np.asarray(static_baseline_raw, dtype=np.float64),
        scaler=static_scaler,
        model=model,
        device=device,
        context_x=context_x,
        future_known=future_known,
        qoi_fn=qoi_fn,
        out_dir=figure_dir,
        batch_size=int(config.get("sobol", {}).get("batch_size", 512)),
        param_names=param_names,
    )

    logging.info("Saved results to %s and figures to %s", result_dir, figure_dir)
