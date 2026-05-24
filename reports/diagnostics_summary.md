# TST Diagnostics Summary

Source: `reports/tst_diagnostics.csv`. One row per final epoch for each dataset/seed/model variant run.

## Status Counts

| Status | Dataset/Variant Groups |
| --- | --- |
| early_stopped_cleanly | 26 |
| underfit | 2 |

## Final-Epoch Aggregate

| Dataset | Task | Family | Variant | Runs | Status Counts | Mean Final Val Metric | Mean Best Epoch | Mean Train-Val Gap | Mean Gate Sparsity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_axis_aligned | classification | ablation | TST-v0 | 3 | early_stopped_cleanly:3 | 0.900813 | 1.000000 | 0.000924 |  |
| synthetic_axis_aligned | classification | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:3 | 0.900813 | 1.000000 | 0.000924 | 0.000000 |
| synthetic_axis_aligned | classification | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:3 | 0.900813 | 1.000000 | 0.000924 | 0.000000 |
| synthetic_axis_aligned | classification | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:3 | 0.900813 | 1.000000 | 0.000924 | 0.000000 |
| synthetic_irrelevant_noise | classification | ablation | TST-v0 | 3 | early_stopped_cleanly:3 | 0.491057 | 1.333333 | 0.000800 |  |
| synthetic_irrelevant_noise | classification | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:3 | 0.491057 | 1.333333 | 0.000800 | 0.000000 |
| synthetic_irrelevant_noise | classification | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:3 | 0.510569 | 2.333333 | -0.002426 | 0.000000 |
| synthetic_irrelevant_noise | classification | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:3 | 0.508943 | 1.000000 | -0.000800 | 0.000000 |
| synthetic_piecewise | regression | ablation | TST-v0 | 3 | early_stopped_cleanly:2, underfit:1 | 0.804367 | 8.333333 | -0.019458 |  |
| synthetic_piecewise | regression | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:2, underfit:1 | 0.804387 | 8.333333 | -0.019494 | 0.000000 |
| synthetic_piecewise | regression | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:2, underfit:1 | 0.803567 | 8.000000 | -0.020271 | 0.000000 |
| synthetic_piecewise | regression | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:3 | 0.803839 | 5.000000 | -0.020232 | 0.000000 |
| synthetic_regime | regression | ablation | TST-v0 | 3 | early_stopped_cleanly:1, overfit:1, underfit:1 | 0.673953 | 8.000000 | 0.048120 |  |
| synthetic_regime | regression | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:1, overfit:1, underfit:1 | 0.674320 | 7.666667 | 0.048102 | 0.000000 |
| synthetic_regime | regression | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:2, underfit:1 | 0.672374 | 6.666667 | 0.043756 | 0.000000 |
| synthetic_regime | regression | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:2, underfit:1 | 0.639333 | 9.000000 | 0.046634 | 0.000000 |
| synthetic_rotated | classification | ablation | TST-v0 | 3 | early_stopped_cleanly:3 | 0.624390 | 1.000000 | 0.001560 |  |
| synthetic_rotated | classification | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:3 | 0.624390 | 1.000000 | 0.001560 | 0.000000 |
| synthetic_rotated | classification | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:3 | 0.624390 | 1.333333 | 0.001560 | 0.000000 |
| synthetic_rotated | classification | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:3 | 0.624390 | 1.000000 | 0.001560 | 0.000000 |
| synthetic_sparse_high_order | classification | ablation | TST-v0 | 3 | early_stopped_cleanly:3 | 0.972358 | 1.000000 | 0.001584 |  |
| synthetic_sparse_high_order | classification | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:3 | 0.972358 | 1.000000 | 0.001584 | 0.000000 |
| synthetic_sparse_high_order | classification | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:3 | 0.972358 | 1.000000 | 0.001584 | 0.000000 |
| synthetic_sparse_high_order | classification | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:3 | 0.972358 | 1.000000 | 0.001584 | 0.000000 |
| synthetic_xor | classification | ablation | TST-v0 | 3 | early_stopped_cleanly:3 | 0.515447 | 1.333333 | 0.004640 |  |
| synthetic_xor | classification | ablation | TST-v1-Gate | 3 | early_stopped_cleanly:3 | 0.517073 | 1.333333 | 0.001928 | 0.000000 |
| synthetic_xor | classification | ablation | TST-v2-GateFourier | 3 | early_stopped_cleanly:3 | 0.521951 | 1.000000 | -0.001321 | 0.000000 |
| synthetic_xor | classification | ablation | TST-v3-MoE | 3 | early_stopped_cleanly:3 | 0.521951 | 1.333333 | -0.001321 | 0.000000 |

## Reading Notes

- `early_stopped_cleanly` means validation checkpointing worked without numerical instability; it does not mean the model learned a useful boundary.
- Flat metrics with best epoch near 1 are best interpreted as under-training or an inductive-bias mismatch, especially on rotated and irrelevant-noise tasks.
- Gate sparsity is only populated for variants with a gate; blank cells mean the model has no gate rather than zero sparsity.
