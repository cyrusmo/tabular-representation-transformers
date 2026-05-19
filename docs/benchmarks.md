# Benchmarks

Required comparisons: linear/logistic/ridge, MLP, random forest, XGBoost or LightGBM, FT-Transformer or TabTransformer, TabPFN where scale allows, and ablations of Tabular State Transformer blocks.

## Synthetic Stress Track

Run the quick two-dataset smoke benchmark:

```bash
venv/bin/python scripts/run_benchmark.py --suite synthetic --n-samples 512 --max-epochs 2
```

Run the broader synthetic stress suite across repeated seeds:

```bash
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic_stress \
  --seeds 42,43,44 \
  --n-samples 512 \
  --max-epochs 2 \
  --output reports/benchmark_results.md \
  --output-csv reports/benchmark_results.csv
```

The stress suite covers axis-aligned thresholds, XOR interactions, piecewise regression,
irrelevant high-dimensional noise, rotated features, regime switching, and sparse high-order
interactions. Optional tree baselines report dependency errors until benchmark extras are installed.
