# Reports index

Research artifacts live here. Each experiment folder uses the same filenames:

| File | Produced by |
| --- | --- |
| `results.md` / `results.csv` | `scripts/run_benchmark.py` |
| `diagnostics.csv` | `scripts/run_benchmark.py` (TST runs) |
| `summary.md` | `scripts/summarize_benchmark.py` |

Smoke runs belong under `outputs/smoke/` (see `docs/benchmarks.md`).

## Canonical experiments

| Experiment | Path | Description |
| --- | --- | --- |
| Fair comparison | `experiments/fair_comparison/` | Full synthetic-stress suite, seeds 42–44, tuned TST mode |
| Targeted training | `experiments/targeted_training/` | XOR + irrelevant-noise diagnostic, seeds 42–44 |
| OpenML | `experiments/openml/` | Real-data smoke / subset runs |
| Legacy | `experiments/legacy/` | Original default synthetic-stress benchmark (May 2026) |

## Analysis (human-written)

| Document | Path |
| --- | --- |
| TST failure diagnosis | `analysis/tst_failure_diagnosis.md` |
| Diagnostics rollup | `analysis/diagnostics_summary.md` |
| Research narrative (local) | `analysis/research_narrative.md` |

## Archive

Superseded runs are kept under `archive/` with a date | smoke label, e.g.
`archive/targeted_training_2026-05-24_smoke/`.

## Figures

Static topology and decision-surface figures: `figures/` (see `docs/tensor_topology.md`).

3D inductive-bias figures live under `figures/3d/`: the decision-landscape hero grid, per-task
orbit GIFs, the gate bar field, the bias atlas, exported `.npz` meshes (for the future Neural
ODE/SDE fork to re-render without re-fitting), and optional plotly HTML. Regenerate with:

```bash
venv/bin/python scripts/visualize_topology.py --mode 3d-surfaces --output-dir reports/figures
venv/bin/python scripts/visualize_topology.py --mode bias-atlas --output-dir reports/figures
```

## Regenerate

```bash
# Fair comparison (full synthetic stress)
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic_stress \
  --benchmark-mode tuned_tst_benchmark \
  --neural-baselines ft_transformer \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20 \
  --output reports/experiments/fair_comparison/results.md \
  --output-csv reports/experiments/fair_comparison/results.csv \
  --diagnostics-output reports/experiments/fair_comparison/diagnostics.csv

venv/bin/python scripts/summarize_benchmark.py \
  --input reports/experiments/fair_comparison/results.csv \
  --output-md reports/experiments/fair_comparison/summary.md

# Targeted XOR / noise diagnostic
venv/bin/python scripts/run_benchmark.py \
  --suite synthetic_stress \
  --datasets synthetic_xor,synthetic_irrelevant_noise \
  --benchmark-mode tuned_tst_benchmark \
  --models TST-v0,TST-v1-Gate,TST-v3-MoE \
  --neural-baselines ft_transformer \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 50 \
  --tuning-max-epochs 50 \
  --output reports/experiments/targeted_training/results.md \
  --output-csv reports/experiments/targeted_training/results.csv \
  --diagnostics-output reports/experiments/targeted_training/diagnostics.csv
```

More commands: `docs/benchmarks.md`, `docs/experimental_protocol.md`.
