# Experimental Protocol

This project evaluates whether feature-preserving neural tabular models can become competitive on
high-dimensional, noisy, interaction-heavy synthetic tasks.

## Benchmark Scope

The current synthetic stress suite uses:

- axis-aligned threshold classification
- XOR interaction classification
- piecewise non-smooth regression
- irrelevant-feature noise classification
- rotated-feature classification
- regime-switching regression
- sparse high-order interaction classification

The first credible small run uses seeds `42,43,44`, `n_samples=1024`, and `max_epochs=20` for TST
variants. Smoke outputs are written under `outputs/smoke/`; research outputs are written under
`reports/`.

## Models

Baselines:

- Linear/Ridge
- Random Forest
- MLP
- LightGBM
- CatBoost

TST ablations:

- TST-v0
- TST-v1-Gate
- TST-v2-GateFourier
- TST-v3-MoE

XGBoost is not part of the default run on this Python 3.13 environment because its current wheel can
segfault in native data handling. It remains available for isolated compatibility checks.

## Metrics

Classification tasks report accuracy. Regression tasks report RMSE. Each benchmark row records
dataset, seed, task, family, model, variant, status, metric, score, fit time, prediction time, sample
count, feature count, artifact path, and error message.

## Topology Artifacts

Tensor topology figures live under `reports/figures/` and are documented in
`docs/tensor_topology.md`. Decision-surface plots are limited to true two-feature synthetic task
configs and are not projections of higher-dimensional tasks.

## Limitations

These results are first-pass evidence, not final paper-grade claims. Hyperparameter tuning is still
minimal, OpenML coverage is not yet complete, and public superiority claims should be avoided unless
future runs consistently support them against strong tree baselines.
