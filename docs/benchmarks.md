# Benchmarks

Required comparisons: linear/logistic/ridge, MLP, random forest, XGBoost or LightGBM, FT-Transformer or TabTransformer, TabPFN where scale allows, and ablations of Tabular State Transformer blocks.

## Synthetic Stress Track

Run the fast synthetic-stress smoke benchmark:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic_stress \
  --seeds 42,43,44 \
  --n-samples 512 \
  --max-epochs 2 \
  --output outputs/smoke/smoke_benchmark_results.md \
  --output-csv outputs/smoke/smoke_benchmark_results.csv \
  --diagnostics-output outputs/smoke/smoke_tst_diagnostics.csv
```

Run the broader synthetic stress suite across repeated seeds:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic_stress \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20 \
  --output reports/benchmark_results.md \
  --output-csv reports/benchmark_results.csv \
  --diagnostics-output reports/tst_diagnostics.csv
```

Run the local FT-Transformer-style smoke path:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic \
  --datasets synthetic_xor \
  --seeds 42 \
  --n-samples 64 \
  --max-epochs 1 \
  --neural-baselines ft_transformer \
  --output outputs/smoke/ft_transformer_smoke.md \
  --output-csv outputs/smoke/ft_transformer_smoke.csv \
  --diagnostics-output outputs/smoke/ft_transformer_smoke_diagnostics.csv
```

Run the tuned-TST smoke path:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic \
  --datasets synthetic_xor \
  --seeds 42 \
  --n-samples 64 \
  --max-epochs 1 \
  --tuning-max-epochs 2 \
  --benchmark-mode tuned_tst_benchmark \
  --output outputs/smoke/tuned_tst_smoke.md \
  --output-csv outputs/smoke/tuned_tst_smoke.csv \
  --diagnostics-output outputs/smoke/tuned_tst_smoke_diagnostics.csv
```

Future full fair-comparison artifacts should use the synthetic-stress suite and write to
`reports/fair_comparison_results.{md,csv}` plus `reports/fair_comparison_diagnostics.csv`.

The stress suite covers axis-aligned thresholds, XOR interactions, piecewise regression,
irrelevant high-dimensional noise, rotated features, regime switching, and sparse high-order
interactions. Optional tree baselines report explicit `status=error` and `error_message`
values until benchmark extras are installed.

The default benchmark uses safe in-process tree baselines: Random Forest, sklearn Gradient Boosting,
LightGBM, and CatBoost. It excludes XGBoost on this Python 3.13 environment because the current wheel
can segfault in native data handling. XGBoost remains available through `--baselines xgboost` for
isolated compatibility checks; a future runner should execute it in a subprocess so native crashes
become captured failed rows instead of killing the whole benchmark.

`tuned_tst_benchmark` is opt-in. It keeps tree defaults unchanged, emits untuned TST rows, and adds
TST tuned rows selected from the fixed learning-rate budget `1e-4,3e-4,1e-3`. Tuned rows record
`candidate_config_count=3`, `candidate_lrs="1e-4,3e-4,1e-3"`, `tuning_budget_type="lr_only"`, and
selection metadata for auditability.

`--neural-baselines ft_transformer` is also opt-in. It runs a dependency-free local
FT-Transformer-style baseline; it is not a validated reference-paper implementation.

Generate topology figures after a benchmark run:

```bash
venv/bin/python scripts/visualize_topology.py \
  --results-csv reports/benchmark_results.csv \
  --output-dir reports/figures \
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20 \
  --benchmark-report reports/benchmark_results.md
```

Decision-surface figures are limited to true two-feature synthetic task configs.
