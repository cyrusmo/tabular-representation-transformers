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
  --output reports/experiments/legacy/results.md \
  --output-csv reports/experiments/legacy/results.csv \
  --diagnostics-output reports/experiments/legacy/diagnostics.csv
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

Fair-comparison artifacts use the synthetic-stress suite and write to
`reports/experiments/fair_comparison/` (`results.{md,csv}`, `diagnostics.csv`, `summary.md`).

Targeted XOR/noise diagnostic runs write to `reports/experiments/targeted_training/`.
OpenML runs write to `reports/experiments/openml/`.

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

Run the post-P1 architecture probe smoke path:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic \
  --datasets synthetic_xor \
  --seeds 42 \
  --n-samples 64 \
  --max-epochs 1 \
  --baselines mlp,lightgbm \
  --models TST-v4-CLS,TST-v4-Attention,TST-v5-CLS-Cross \
  --output outputs/smoke/architecture_probe_smoke.md \
  --output-csv outputs/smoke/architecture_probe_smoke.csv \
  --diagnostics-output outputs/smoke/architecture_probe_smoke_diagnostics.csv
```

Run the targeted architecture probe only after the diagnostics gate is passing:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic \
  --datasets synthetic_xor,synthetic_irrelevant_noise \
  --seeds 42 \
  --n-samples 512 \
  --max-epochs 50 \
  --baselines mlp,lightgbm \
  --models TST-v1-Gate,TST-v2-GateFourier,TST-v4-CLS,TST-v4-Attention,TST-v5-CLS-Cross \
  --output reports/architecture_probe_results.md \
  --output-csv reports/architecture_probe_results.csv \
  --diagnostics-output reports/architecture_probe_diagnostics.csv
```

Architecture-probe models are explicit-only labels. They do not change the default TST-v0 through
TST-v3 benchmark set.

Run the trainability audit before adding more architecture variants:

```bash
venv/bin/python scripts/run_trainability_audit.py \
  --tasks xor_2f,xor_20f,irrelevant_noise_100f \
  --models mlp,TST-v0,TST-v1-Gate,TST-v4-CLS,TST-v5-CLS-Cross \
  --seeds 42 \
  --n-samples 512 \
  --max-epochs 300 \
  --output reports/trainability_audit_results.md \
  --output-csv reports/trainability_audit_results.csv \
  --diagnostics-output reports/trainability_audit_diagnostics.csv
```

This audit disables TST early stopping and dropout. It is a trainability check, not a benchmark
claim: the decision point is whether TST can memorize no-noise 2-feature XOR and whether failure
begins only after irrelevant features are added. Recorded Path B interpretation lives in
`reports/analysis/research_narrative.md` next to the XOR failure.

Generate topology figures after a benchmark run:

```bash
venv/bin/python scripts/visualize_topology.py \
  --results-csv reports/experiments/legacy/results.csv \
  --output-dir reports/figures \
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20 \
  --benchmark-report reports/experiments/legacy/results.md
```

Decision-surface figures are limited to true two-feature synthetic task configs.
