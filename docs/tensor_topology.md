# Tensor Topology

These figures make the Tabular State Transformer representation legible. They are research
artifacts, not claims of model superiority.

## Figures

- `reports/figures/tensor_topology_schematic.png`: raw row to feature tokens, gates, spectral expansion, interaction block, pooled state, and head.
- `reports/figures/representation_shape_diagram.png`: tensor shape flow from `Batch x Features` to `Batch x d_model`.
- `reports/figures/feature_gate_heatmap.png`: learned sparse-gate values for TST-v1/TST-v2 on synthetic stress tasks.
- `reports/figures/synthetic_task_topology_grid.png`: decision surfaces for true 2D synthetic tasks only.

## Generation

```bash
venv/bin/python scripts/visualize_topology.py \
  --results-csv reports/experiments/legacy/results.csv \
  --output-dir reports/figures \
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20
```

Gate values are extracted through `extract_gate_values(model, X_valid)`: global gates use
`sigmoid(logits)`, input-dependent gates can expose or cache validation-set activations, and missing
gates are represented as `NaN`.

Decision surfaces are intentionally limited to native two-feature tasks. Higher-dimensional tasks are
not projected into 2D here, because that would make the visualization look more faithful than it is.
