# Research Narrative (Path B)

**Date:** 2026-08-05  
**Decision:** Path B — publish the ablation/diagnostic harness and the negative result. Do not claim TST competitiveness. Freeze the default model ladder (v0–v3); do not scale failed v4/v5 probes.

## Thesis tested

Feature identity + sparse gates + spectral/interaction blocks can make neural tabular models competitive on interaction-heavy and high-dimensional noise tasks relative to strong tree and MLP baselines.

Under this repository’s protocols, that thesis is **falsified**.

## Evidence reviewed

| Artifact | Role |
| --- | --- |
| `reports/experiments/fair_comparison/` | Full synthetic-stress suite, seeds 42–44, `n_samples=1024`, `max_epochs=20`, LR-only tuned TST + local FT-Transformer-style baseline |
| `reports/experiments/targeted_training/` | XOR + irrelevant-noise, seeds 42–44, `max_epochs=50` |
| `reports/architecture_probe_results.md` | Post-P1 CLS / attention / feature-cross probe (research gate: **failed**) |
| `reports/experiments/openml/` | Real-data **smoke** only (1 seed, `n=512`) — not paper-grade |
| `reports/analysis/tst_failure_diagnosis.md` | Legacy diagnostics: early-stop “clean” ≠ learned; gates historically stuck near `sigmoid(1)` |
| `reports/trainability_audit_results.md` | Memorization audit (2-feature XOR → 20-feature XOR → irrelevant noise) |

Summaries in each experiment folder were regenerated from the corresponding `results.csv` on 2026-08-05.

## Headline results

### Fair comparison (seeds 42–44)

From `reports/experiments/fair_comparison/results.md`:

- Tree mean rank: **2.655**
- Untuned TST mean rank: **9.381**
- Tuned TST mean rank: **8.298**
- Tuning helps (~1 mean-rank point) but does **not** close the tree gap.

Decision tasks (mean test accuracy over seeds):

| Task | LightGBM / trees | MLP | Best tuned TST |
| --- | ---: | ---: | ---: |
| `synthetic_xor` | ~0.99 | ~0.77 | ~0.52 (chance) |
| `synthetic_irrelevant_noise` | ~0.93 | ~0.81 | ~0.63 |
| `synthetic_rotated` | ~0.96 | ~0.95 | ~0.93 (tuned v0) |
| `synthetic_axis_aligned` | ~1.00 | ~0.93 | ~0.90 |
| `synthetic_sparse_high_order` | ~1.00 | ~0.97 | ~0.97 |

Competitive niches exist on easy / sparse-high-order structure. They do not salvage the thesis on XOR or junk-feature noise.

### Targeted training (50 epochs)

| Task | LightGBM | MLP | Best tuned TST |
| --- | ---: | ---: | ---: |
| `synthetic_xor` | ~0.99 | ~0.77 | ~0.52 — **bar to beat MLP not met** |
| `synthetic_irrelevant_noise` | ~0.93 | ~0.81 | ~0.85 (v1-Gate-tuned) — partial recovery, still below trees |

Longer training + gate optimizer fixes help noise selection somewhat. They do not fix XOR.

### Architecture probe

CLS pooling, attention pooling, and first-k pairwise cross tokens did not improve the primary metric on XOR/noise (`reports/architecture_probe_results.md`). Variants remain near chance and must **not** be scaled to a full fair comparison.

### OpenML

Smoke plumbing only. Treat as non-claim evidence. Competitive only on `credit-g` in that small run; elsewhere TST trails LightGBM/MLP. Expand only if a follow-up study needs real-data breadth — not required for Path B.

## Trainability audit

Artifact: `reports/trainability_audit_results.md`  
Settings: seed 42, `n_samples=512`, dropout `0`, early stopping **disabled**, `max_epochs=300`, audit XOR uses `noise=0.0`.

| Task | Best TST train acc | Best TST test acc | MLP test |
| --- | ---: | ---: | ---: |
| 2-feature XOR | **1.000** | ~0.97–0.99 | ~0.99 |
| 20-feature XOR (`noise=0`) | **1.000** | ~0.83–0.98 | ~0.81 |
| Irrelevant noise (100f) | **1.000** | ~0.72–0.97 | ~0.74 |

Automatic verdict from the audit script:

> TST can memorize the audit tasks; prioritize generalization and regularization diagnostics.

**Interpretation next to the XOR failure:** under a long, no-early-stop budget TST is **not** incapable of fitting these synthetic labels. The published fair/targeted protocol (`max_epochs` 20–50, early stopping on) still yields **~chance test accuracy on 20-feature XOR**. That gap points to **training protocol / early-stop / generalization**, not “add another module.” Path B still freezes the architecture ladder; the only optional digression is training-budget diagnostics, not v4/v5 scaling.

## What we learned

1. **The contribution is the harness, not a SOTA model.** Multi-seed suites, gate diagnostics, early-stop honesty checks, LR-only fair tuning, a local FT-Transformer-style baseline, and topology figures make the failure reproducible.
2. **“Clean” early stopping ≠ learning.** Hard-task runs can stop with chance validation accuracy while being labeled `early_stopped_cleanly`. The trainability audit shows the same architecture can memorize XOR when early stopping is off and the epoch budget is large — so published chance-level XOR scores are a **protocol / generalization** failure mode, not proof the network cannot fit the labels.
3. **Module stacking without trainability is a trap.** Ablations often score-identical on hard tasks under short budgets; gates historically never sparsified until `gate_init` / `gate_lr_multiplier`. After those fixes, noise improved under longer budgets; fair/targeted XOR still fails.
4. **Inductive bias appears where the architecture matches the task.** Token + mean-pool looks sane on axis-aligned / sparse-high-order; XOR and junk features expose the representation.
5. **LR-only tuning does not close a tree gap.** More of the same search is unlikely to rescue XOR.
6. **Finance framing stays non-claim.** See `docs/finance_disclaimer.md` — architecture demos, not alpha.

## Frozen model ladder

Default benchmark / public story:

- **In scope:** TST-v0, TST-v1-Gate, TST-v2-GateFourier, TST-v3-MoE (plus baselines and optional local FT-Transformer-style).
- **Frozen / opt-in only:** TST-v4-CLS, TST-v4-Attention, TST-v5-CLS-Cross — documented failures; do not expand to full suites.
- **Out of Path B scope:** wavelet path (implemented, unmeasured), Hub mixin polish, Neural ODE/SDE / Track A continuity — separate forks if pursued ([`docs/tensor_topology.md`](../../docs/tensor_topology.md)).

## What not to try next (for this thesis)

- Scaling v4/v5 to fair comparison
- Adding more heads/blocks without a passing trainability audit and a primary-metric gain on XOR/noise
- Public superiority claims or trading / alpha framing
- Treating OpenML smoke as paper-grade evidence

## Optional follow-ups (explicitly not required for Path B)

- Training-protocol digression only: why early-stopped short budgets stay at chance on XOR while the 300-epoch audit memorizes (patience, LR schedule, noise setting). Do **not** use this as a license to stack modules.
- If a later paper needs real-data breadth: multi-seed OpenML with honest “smoke vs full” labeling.
- Wavelet ablation under the same protocol **or** remove wavelet from the architecture story.

## One-sentence contribution

This repository is a reproducible demonstration that the tested feature-token + gate/Fourier/MoE inductive biases do **not** beat trees on the stated hard synthetic stress suite, plus a diagnostic harness that makes that failure hard to hide.
