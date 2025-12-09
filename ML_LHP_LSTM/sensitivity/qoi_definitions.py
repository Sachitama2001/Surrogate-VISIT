"""Quantity-of-Interest (QoI) helpers for sensitivity analysis."""
from __future__ import annotations

from typing import Callable, Dict
import torch


def _validate_indices(start_day: int, end_day: int, flux_index: int) -> None:
    if not (0 <= start_day <= end_day <= 29):
        raise ValueError("start_day/end_day must satisfy 0 <= start <= end <= 29")
    if flux_index < 0 or flux_index > 3:
        raise ValueError("flux_index must be in [0, 3] for GPP, NPP, ER, NEP")


def _mean_flux(pred: torch.Tensor, flux_index: int, start_day: int, end_day: int) -> torch.Tensor:
    _validate_indices(start_day, end_day, flux_index)
    window = pred[:, start_day : end_day + 1, flux_index]
    return window.mean(dim=1)


def _cumulative_flux(pred: torch.Tensor, flux_index: int, start_day: int, end_day: int) -> torch.Tensor:
    _validate_indices(start_day, end_day, flux_index)
    window = pred[:, start_day : end_day + 1, flux_index]
    return window.sum(dim=1)


def get_qoi_function(name: str, options: Dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return a function that maps model outputs (batch, 30, 4) -> (batch,).
    """
    name = name.lower()
    flux_index = int(options.get("flux_index", 3))
    start_day = int(options.get("start_day", 0))
    end_day = int(options.get("end_day", 29))

    if name in {"mean_nep_30d", "mean_nep"}:
        return lambda pred: _mean_flux(pred, flux_index=3, start_day=start_day, end_day=end_day)
    if name in {"cumulative_nep_30d", "cum_nep"}:
        return lambda pred: _cumulative_flux(pred, flux_index=3, start_day=start_day, end_day=end_day)
    if name in {"mean_gpp_30d", "mean_gpp"}:
        return lambda pred: _mean_flux(pred, flux_index=0, start_day=start_day, end_day=end_day)
    if name == "mean_flux":
        return lambda pred: _mean_flux(pred, flux_index=flux_index, start_day=start_day, end_day=end_day)
    if name == "cumulative_flux":
        return lambda pred: _cumulative_flux(pred, flux_index=flux_index, start_day=start_day, end_day=end_day)

    raise KeyError(f"Unknown QoI name: {name}")
