# TST Failure Diagnosis

**Source:** `reports/experiments/legacy/diagnostics.csv` (synthetic stress, seeds 42–44, 1024 samples, default training)  
**Analysis date:** 2026-05-24

## Executive summary

TST variants fail on **interaction-heavy** and **high-dimensional noise** tasks while matching each other almost exactly. Gates never sparsify (`gate_sparsity = 0.0` everywhere). Hard tasks show **chance-level validation accuracy** with **early stopping at epoch 1–3**, classified mostly as `early_stopped_cleanly` rather than `underfit` because train metrics briefly cross the diagnostic “poor” threshold.

## Cluster 1: `effective_training_status`

| Status | Runs (final epoch) | Share |
| --- | ---: | ---: |
| `early_stopped_cleanly` | 75 | 88% |
| `underfit` | 7 | 8% |
| `overfit` | 2 | 2% |
| `no_learning` / `unstable` | 0 | 0% |

**Interpretation:** Training is numerically stable but **stops before useful validation gains** on hard tasks. The heuristic labels many XOR/noise runs as “clean” early stops because validation hovers near 0.5 (random) and patience triggers quickly.

## Cluster 2: Gate sparsity (v1–v3 only)

| Metric | Value |
| --- | --- |
| `gate_sparsity` (fraction of gates &lt; 0.1) | **0.0** on all 63 gated runs |
| `gate_mean` | **0.729–0.731** (constant across epochs) |
| `has_gate` | Matches `sigmoid(1.0) ≈ 0.731` from default gate init |

**Root cause (mechanism):**

1. The committed diagnostics were generated before the gate fix, when gated configs started around `sigmoid(1.0) ≈ 0.73`.
2. Even after moving the default gate init to `0.0`, ordinary AdamW updates at the model LR were too slow: a focused test with `gate_l1=0.01` still produced `gate_sparsity = 0.0`.
3. Sparsity metric requires values **&lt; 0.1**; sigmoid-logit gates need either lower initialization, a stronger gate optimizer step, or both to reach that threshold in short tabular runs.

**Fix applied:** Add explicit `gate_init` and `gate_lr_multiplier` config fields, initialize gated ablations at `gate_init: -1.0`, and optimize gate logits with a larger LR multiplier. A focused test verifies that strong L1 can now drive nonzero sparsity on `synthetic_irrelevant_noise`.

## Cluster 3: Hard-task failures

### `synthetic_xor` (20 features, XOR label)

| Model | Mean best val acc | Typical best epoch | Test acc (benchmark) |
| --- | ---: | ---: | ---: |
| TST v0–v3 | ~0.51 | 1–2 | ~0.51 |
| MLP | — | — | **0.77** |
| LightGBM / CatBoost | — | — | **0.99** |

- All TST ablations **identical** (no gate/Fourier/MoE benefit).
- Diagnostics: val acc plateaus at ~0.51; `prediction_std` collapses → near-constant predictions.
- **Conclusion:** Representation/optimization limit (mean-pooled tokens, shallow interaction), not missing gate sparsity alone.

### `synthetic_irrelevant_noise` (100 features, few signal)

| Model | Mean best val acc | Gate mean | Gate sparsity |
| --- | ---: | ---: | ---: |
| TST v0–v3 | ~0.51 | 0.731 (gated) | 0.0 |
| LightGBM / CatBoost | — | — | **0.92** test |

- TST at **chance**; trees strong → feature selection/noise rejection works for trees, not current TST.
- Gates do not down-weight irrelevant dimensions.

### Tasks where TST is competitive

- `synthetic_axis_aligned`, `synthetic_sparse_high_order`: TST ~0.91–0.97 vs trees (easy structure aligned with token + mean pool).
- `synthetic_rotated`: TST ~0.62 vs trees ~0.93–0.96 (partial failure).
- Regression tasks (`piecewise`, `regime`): TST underperforms trees; v3-MoE slightly better on `regime` only.

## Cross-cutting patterns

1. **Ablation equivalence:** v0–v3 match on XOR, irrelevant-noise, and rotated → added blocks inactive or ineffective under current training.
2. **Early stopping vs capacity:** Best epoch 1–3 on hard tasks; increasing patience alone unlikely to fix without architectural/training changes.
3. **Diagnostics vs benchmark:** Final-epoch status understates failure; use **best_val_metric** and **test scores** for decisions.

## Actionable conclusions (priority)

| Priority | Action | Status (2026-08-05) |
| --- | --- | --- |
| P0 | Re-run gated benchmarks after gate init + gate LR fix | Done via fair comparison + targeted training |
| P0 | Run `tuned_tst_benchmark` + FT-Transformer baseline | Done — `reports/experiments/fair_comparison/` |
| P1 | Targeted XOR/noise: longer epochs, LR sweep | Done — XOR still fails the MLP bar (~0.52 vs ~0.77) |
| P1 | OpenML subset beyond `adult` | Smoke only (`reports/experiments/openml/`); not paper-grade |
| P2 | Persist checkpoints + preprocessor | Done for TST benchmark artifacts |
| P2 | Path B narrative if fair + targeted still fail | **Chosen** — see `reports/analysis/research_narrative.md` |

## Recommended experiments (historical)

These were the next commits after the 2026-05-24 diagnosis; they are now complete or superseded by Path B:

1. `gate_l1` sweep / gate optimizer fix validation.
2. Fair comparison: `synthetic_stress`, seeds 42–44, `max_epochs=20`, `tuned_tst_benchmark`, `ft_transformer`.
3. Targeted: XOR + irrelevant_noise only, `max_epochs=50`, LR grid after gate fix.

Do not stack further modules without a passing trainability audit and a primary-metric gain on XOR.
