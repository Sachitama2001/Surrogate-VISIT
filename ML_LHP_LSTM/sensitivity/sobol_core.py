"""Core utilities for Sobol-based global sensitivity analysis."""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from SALib.analyze import sobol
from SALib.sample import saltelli

from model import create_model
from dataset import VISITTimeSeriesDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def load_bounds(range_csv: Path) -> Dict[str, Dict[str, float]]:
    if not range_csv.exists():
        raise FileNotFoundError(range_csv)

    bounds: Dict[str, Dict[str, float]] = {}
    raw_lines = range_csv.read_text().splitlines()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.lower().startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        idx_name, min_str, max_str = parts[:3]
        name_match = re.search(r"\"([^\"]+)\"", idx_name)
        if name_match:
            name = name_match.group(1)
        else:
            name = idx_name.split(":", 1)[-1].strip().strip('"')
        min_val = _parse_float(min_str)
        max_val = _parse_float(max_str)
        max_ref = None if max_val is not None else max_str.strip().strip('"')
        bounds[name] = {"min": min_val, "max": max_val, "max_ref": max_ref}

    def resolve_max(param: str, stack: set[str]) -> float:
        entry = bounds[param]
        if entry.get("max") is not None or entry.get("max_ref") is None:
            return entry["max"]
        ref = entry["max_ref"]
        if ref not in bounds:
            raise KeyError(f"max reference '{ref}' for parameter '{param}' not found in range file")
        if ref in stack:
            raise ValueError(f"Circular reference detected while resolving '{param}' -> '{ref}'")
        stack.add(param)
        resolved = resolve_max(ref, stack)
        stack.remove(param)
        if resolved is None:
            raise ValueError(f"Unable to resolve max value for parameter '{param}' via reference '{ref}'")
        entry["max"] = resolved
        return resolved

    for name in list(bounds.keys()):
        resolve_max(name, set())

    for name, entry in bounds.items():
        if entry.get("min") is None or entry.get("max") is None:
            raise ValueError(f"Bounds missing numeric min/max for parameter '{name}' in {range_csv}")

    return bounds


def load_model_and_scalers(config: dict):
    model_cfg = config.get("model", {})
    checkpoint_path = Path(model_cfg.get("checkpoint_path", "artifacts/checkpoint_best.pt"))
    config_path = Path(model_cfg.get("config_path", checkpoint_path.with_name("config.json")))
    device = model_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU")
        device = "cpu"

    import pickle

    with open(Path(config["scalers"]["dynamic"]), "rb") as f:
        dynamic_scaler = pickle.load(f)
    with open(Path(config["scalers"]["static"]), "rb") as f:
        static_scaler = pickle.load(f)

    dynamic_dim = int(len(dynamic_scaler.mean_))
    static_dim = int(len(static_scaler.mean_))

    train_cfg = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            train_cfg = json.load(f)

    hidden_size = int(train_cfg.get("hidden_size", 256))
    num_layers = int(train_cfg.get("num_layers", 2))
    dropout = float(train_cfg.get("dropout", 0.1))
    output_dim = int(train_cfg.get("output_dim", 4))

    model = create_model(
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    logging.info("Loaded model from %s on device %s", checkpoint_path, device)
    return model, static_scaler, dynamic_scaler, device, train_cfg


def load_reference_window(config: dict, static_scaler, dynamic_scaler, device: str, train_cfg: dict):
    window_cfg = config.get("window", {})
    sample_id = int(window_cfg.get("sample_id", 0))
    window_index = int(window_cfg.get("window_index", 0))
    split = window_cfg.get("split", "test")

    data_dir = Path(train_cfg.get("data_dir", ""))
    if not data_dir:
        raise ValueError("data_dir missing in training config; cannot load dataset")

    context_len = int(train_cfg.get("context_len", 180))
    prediction_len = int(train_cfg.get("prediction_len", 30))

    dataset = VISITTimeSeriesDataset(
        data_dir=data_dir,
        split=split,
        context_len=context_len,
        prediction_len=prediction_len,
        scaler_dynamic=dynamic_scaler,
        scaler_static=static_scaler,
        fit_scaler=False,
    )

    if sample_id not in dataset.sample_ids:
        raise KeyError(f"sample_id {sample_id} not found in dataset")

    sample_idx = dataset.sample_ids.index(sample_id)
    sample_windows = [(i, w) for i, w in enumerate(dataset.windows) if w["sample_id"] == sample_idx]
    if not sample_windows:
        raise ValueError(f"No windows found for sample_id {sample_id} (split={split})")
    if window_index >= len(sample_windows):
        raise IndexError(
            f"window_index {window_index} out of range for sample_id {sample_id}; total {len(sample_windows)}"
        )

    selected_idx = sample_windows[window_index][0]
    batch = dataset[selected_idx]

    context_x = batch["context_x"].unsqueeze(0).to(device)
    future_known = batch["future_known"].unsqueeze(0).to(device)

    param_df = pd.read_csv(Path(data_dir) / "parameter_summary.csv")
    param_names = list(param_df.columns[1:])
    static_baseline_raw = dataset.static_params[sample_idx]

    logging.info(
        "Loaded reference window: sample_id=%s (index=%s), window_index=%s, split=%s",
        sample_id,
        sample_idx,
        window_index,
        split,
    )

    return context_x, future_known, static_baseline_raw, param_names


def setup_parameter_space(config: dict, param_names: List[str]) -> Tuple[dict, List[int]]:
    params_cfg = config.get("parameters", {})
    range_csv = Path(params_cfg.get("range_csv", "configs/range_of_paras.csv"))
    bounds_lookup = load_bounds(range_csv)

    include_list = params_cfg.get("include") or []
    if len(include_list) == 0:
        include_list = [name for name in param_names if name in bounds_lookup]

    problem_names: List[str] = []
    problem_bounds: List[List[float]] = []
    param_indices: List[int] = []

    for name in include_list:
        if name not in bounds_lookup:
            raise KeyError(f"Parameter '{name}' missing in range file {range_csv}")
        if name not in param_names:
            raise KeyError(f"Parameter '{name}' not found in parameter_summary columns")
        idx = param_names.index(name)
        entry = bounds_lookup[name]
        problem_names.append(name)
        problem_bounds.append([float(entry["min"]), float(entry["max"])])
        param_indices.append(idx)

    problem = {"num_vars": len(problem_names), "names": problem_names, "bounds": problem_bounds}
    logging.info("Parameter space: %d vars from %s", problem["num_vars"], range_csv)
    return problem, param_indices


def evaluate_qoi_with_sobol_samples(
    model: torch.nn.Module,
    device: str,
    static_scaler,
    context_x: torch.Tensor,
    future_known: torch.Tensor,
    problem: dict,
    param_names: List[str],
    param_indices: List[int],
    static_baseline_raw: np.ndarray,
    qoi_fn,
    base_sample_size: int,
    calc_second_order: bool,
    batch_size: int = 512,
):
    if problem["num_vars"] == 0:
        raise ValueError("No parameters selected for sensitivity analysis")

    param_values = saltelli.sample(problem, base_sample_size, calc_second_order=calc_second_order)
    n_total = param_values.shape[0]

    baseline = np.asarray(static_baseline_raw, dtype=np.float64)
    static_matrix = np.repeat(baseline[None, :], n_total, axis=0)
    for col_idx, param_idx in enumerate(param_indices):
        static_matrix[:, param_idx] = param_values[:, col_idx]

    mean = np.asarray(static_scaler.mean_, dtype=np.float64)
    scale = np.asarray(static_scaler.scale_, dtype=np.float64)
    static_scaled = (static_matrix - mean) / scale

    y = np.zeros(n_total, dtype=np.float64)
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            static_batch = torch.from_numpy(static_scaled[start:end]).float().to(device)
            context_batch = context_x.repeat(end - start, 1, 1)
            future_batch = future_known.repeat(end - start, 1, 1)
            pred = model(context_batch, static_batch, future_batch)
            qoi_val = qoi_fn(pred)
            if qoi_val.ndim != 1:
                raise ValueError("QoI function must return shape (batch,)")
            y[start:end] = qoi_val.cpu().numpy()

    logging.info("Evaluated QoI for %d Sobol samples", n_total)
    return param_values, y


def compute_sobol_indices(problem: dict, y: np.ndarray, calc_second_order: bool):
    result = sobol.analyze(problem, y, calc_second_order=calc_second_order, print_to_console=False)
    logging.info("Sobol indices computed (S1/ST)")
    return result
