# Surrogate-VISIT

This repository collects two complementary workflows around the VISIT (Vegetation Integrative Simulator for Trace gases) long-term forest site (LHP):

1. **Parameter screening** elementary effects by Morris 1911.
2. **A deep-learning surrogate (LSTM) for flux emulation and inverse modeling.**

The goal is to automate parameter recalibration against observed fluxes.

## Repository Layout

| Directory | Purpose | Notes |
| --- | --- | --- |
| `LHP_elementary_effects/` | Morris-based screening workspace: SALib design generation, VISIT runners, μ*/σ aggregation, plotting utilities. | `LHP_elementary_effects/README.md`. |
| `ML_LHP_LSTM/` | LSTM surrogate with static-parameter conditioning for GPP/NPP/NEP/Rh multi-step forecasts plus observation-driven inversion. | `ML_LHP_LSTM/README.md`. |
| `LHP_elementary_effects/docs/*`, `ML_LHP_LSTM/docs/*` | Metadata, experiment plans, and reproducibility notes. | Updated frequently as research notes. |


## License

Usage is currently restricted to research purposes (no formal OSS license yet). Contact the maintainer before redistributing code or results outside the project context.
