# Tensor Topology

These figures make the Tabular State Transformer representation legible. They are research
artifacts, not claims of model superiority.

## 2D figures

- `reports/figures/tensor_topology_schematic.png`: raw row to feature tokens, gates, spectral expansion, interaction block, pooled state, and head.
- `reports/figures/representation_shape_diagram.png`: tensor shape flow from `Batch x Features` to `Batch x d_model`.
- `reports/figures/feature_gate_heatmap.png`: learned sparse-gate values for TST-v1/TST-v2 on synthetic stress tasks.
- `reports/figures/synthetic_task_topology_grid.png`: decision surfaces for true 2D synthetic tasks only.

## 3D inductive-bias figures

- `reports/figures/3d/decision_landscape_grid_3d.png`: hero grid. Rows are true-2D synthetic
  tasks, columns are model families (Linear, RandomForest, LightGBM, MLP, FT-Transformer,
  TST-v0..v3), height is the predicted score / class probability. Each family's inductive bias
  should be readable from the surface shape alone: flat tilted planes (linear), axis-aligned
  plateaus and sharp ridges (trees), smooth rotated ramps (MLP), and token-attention surfaces
  (FT / TST ablations).
- `reports/figures/3d/decision_landscape_<task>_orbit.gif`: short orbit render per task row,
  same camera grammar as the static grid.
- `reports/figures/3d/gate_bar_field_3d.png`: intended vs learned per-feature gate sparsity on
  `synthetic_irrelevant_noise` (signal features are `x0, x1, x2`).
- `reports/figures/3d/bias_atlas.png`: layered tensor-topology diagram plus the TST-v0..v3
  module ladder and analytic bias-signature panels (tree partition vs MLP mixing vs TST tokens).
- `reports/figures/3d/meshes/<task>__<model>.npz`: exported meshes (`xx`, `yy`, `zz`,
  `x_train`, `y_train`, `task_type`) so downstream work can re-render without re-fitting.
- `reports/figures/3d/html/decision_landscape_<task>.html`: interactive plotly surfaces,
  generated with `--export-html` (requires the `viz3d` extra).

## Generation

```bash
# Existing 2D figures
venv/bin/python scripts/visualize_topology.py \
  --mode 2d \
  --results-csv reports/experiments/legacy/results.csv \
  --output-dir reports/figures \
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20

# 3D decision landscapes + orbit GIFs + gate bar field (add --export-html for plotly HTML)
venv/bin/python scripts/visualize_topology.py \
  --mode 3d-surfaces \
  --output-dir reports/figures \
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20

# Bias atlas (no model fitting required)
venv/bin/python scripts/visualize_topology.py --mode bias-atlas --output-dir reports/figures

# Everything
venv/bin/python scripts/visualize_topology.py --mode all --output-dir reports/figures
```

Gate values are extracted through `extract_gate_values(model, X_valid)`: global gates use
`sigmoid(logits)`, input-dependent gates can expose or cache validation-set activations, and missing
gates are represented as `NaN`.

## Honesty rules

- Decision surfaces and landscapes are intentionally limited to native two-feature tasks.
  Higher-dimensional tasks are not projected into 2D or 3D here, because that would make the
  visualization look more faithful than it is. High-dimensional stress tasks appear only as
  metric / diagnostic panels (gate bar field, heatmaps, benchmark tables).
- Captions must describe these as research artifacts from an ablation harness, never as
  evidence of model superiority or trading claims (see `docs/finance_disclaimer.md`).

## Future visual continuity

These are separate-track / fork ideas. They do not change the Path B conclusion for this repo
(`reports/analysis/research_narrative.md`).

- Track A bridge: financial-state tensor and graph figures made in the private Track A repo
  should reuse the same camera grammar (`elev=28, azim=-60`), the viridis surface scale, and
  the "bias signature" caption style so cross-track figures read as one research program.
- Neural ODE / SDE fork: continuous-time models live in a separate future fork, not this repo.
  Candidate panels there are trajectory flows and stochastic path ensembles rendered over the
  same synthetic landscapes; the `.npz` mesh exports exist so that fork can re-render Track B
  baselines without re-fitting them.
