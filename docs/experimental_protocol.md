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
- Gradient Boosting
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

## TST Training Diagnostics

TST runs emit `reports/tst_diagnostics.csv` for the credible benchmark. Each row records dataset,
seed, variant, epoch, train loss, train metric, validation metric, train-validation gap, gradient
norm, prediction mean/std, gate summaries when present, best epoch, best validation metric, final
validation metric, final-vs-best gap, early stopping state, and `effective_training_status`.

The `effective_training_status` heuristic is deterministic and intentionally simple:

- `unstable`: any tracked loss, metric, prediction summary, or gradient norm is NaN/inf, average
  gradient norm exceeds `1e4`, or prediction mean/std magnitude exceeds `1e6`.
- `no_learning`: train loss improves by no more than `1e-4` and train metric improves by no more
  than `1e-4`.
- `overfit`: train metric improves by more than `1e-4` while validation metric degrades by more than
  `1e-4`, or the final train-validation gap exceeds `0.15`.
- `underfit`: train and validation remain poor and the best epoch is near the final epoch. Poor means
  accuracy below `0.7` for classification or RMSE above `0.75` for regression. Near-final means the
  best epoch falls in the final 20% of observed epochs. This label is also used as the conservative
  fallback for clean finite runs that did not trigger early stopping.
- `early_stopped_cleanly`: finite metrics, no failure heuristic triggered, and validation-based
  patience stopped training after a prior best checkpoint.

Validation checkpointing uses task-aware direction: classification maximizes accuracy and regression
minimizes RMSE. The returned model restores the best validation checkpoint before benchmark scoring.

## Topology Artifacts

Tensor topology figures live under `reports/figures/` and are documented in
`docs/tensor_topology.md`. Decision-surface plots are limited to true two-feature synthetic task
configs and are not projections of higher-dimensional tasks.

## Limitations

These results are first-pass evidence, not final paper-grade claims. Hyperparameter tuning is still
minimal, OpenML coverage is not yet complete, and public superiority claims should be avoided unless
future runs consistently support them against strong tree baselines.
