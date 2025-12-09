"""Command-line entry point for Sobol sensitivity analysis using the VISIT-LSTM surrogate."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .qoi_definitions import get_qoi_function
from .sobol_core import (
    compute_sobol_indices,
    evaluate_qoi_with_sobol_samples,
    load_model_and_scalers,
    load_reference_window,
    set_seed,
    setup_parameter_space,
)
from .plot_sensitivity import save_results_and_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sobol sensitivity analysis with VISIT-LSTM surrogate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sensitivity/config_sensitivity.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    set_seed(int(config.get("seed", 42)))

    model, static_scaler, dynamic_scaler, device, train_cfg = load_model_and_scalers(config)
    context_x, future_known, static_baseline_raw, param_names = load_reference_window(
        config, static_scaler, dynamic_scaler, device, train_cfg
    )

    qoi_cfg = config.get("qoi", {})
    qoi_fn = get_qoi_function(qoi_cfg.get("name", "mean_nep_30d"), qoi_cfg.get("options", {}))

    problem, param_indices = setup_parameter_space(config, param_names)

    sobol_cfg = config.get("sobol", {})
    param_values, y = evaluate_qoi_with_sobol_samples(
        model=model,
        device=device,
        static_scaler=static_scaler,
        context_x=context_x,
        future_known=future_known,
        problem=problem,
        param_names=param_names,
        param_indices=param_indices,
        static_baseline_raw=static_baseline_raw,
        qoi_fn=qoi_fn,
        base_sample_size=int(sobol_cfg.get("base_sample_size", 1024)),
        calc_second_order=bool(sobol_cfg.get("calc_second_order", True)),
        batch_size=int(sobol_cfg.get("batch_size", 512)),
    )

    sobol_result = compute_sobol_indices(
        problem, y, calc_second_order=bool(sobol_cfg.get("calc_second_order", True))
    )

    s1 = sobol_result.get("S1")
    st = sobol_result.get("ST")
    names = problem.get("names", [])
    if s1 is not None and st is not None:
        top_order = sorted(range(len(names)), key=lambda i: abs(s1[i]), reverse=True)[:5]
        logging.info("Top-5 S1/ST:")
        for i in top_order:
            logging.info("  %s: S1=%.4f, ST=%.4f", names[i], s1[i], st[i])

    save_results_and_figures(
        problem=problem,
        param_values=param_values,
        y=y,
        sobol_result=sobol_result,
        config=config,
        param_names=param_names,
        static_baseline_raw=static_baseline_raw,
        static_scaler=static_scaler,
        qoi_fn=qoi_fn,
        model=model,
        device=device,
        context_x=context_x,
        future_known=future_known,
    )


if __name__ == "__main__":
    main()
