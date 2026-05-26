| Model | Dataset | Seed | Task | Family | Variant | Status | Metric | Score | Fit Seconds | Predict Seconds | N Samples | N Features | Artifact Path | Error Message | Notes | Benchmark Mode | Base Variant | Selected Config Id | Selected Lr | Selection Metric | Selection Mode | Selection Score | Selected Epoch | Candidate Config Count | Candidate Lrs | Tuning Budget Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | synthetic_xor | 42 | classification | baseline | mlp | ok | accuracy | 0.679612 | 0.0415 | 0.0044 | 512 | 20 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_xor | 42 | classification | baseline | lightgbm | ok | accuracy | 0.990291 | 0.1828 | 0.0033 | 512 | 20 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_xor | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.514563 | 5.6827 | 0.014 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v2-GateFourier | synthetic_xor | 42 | classification | ablation | TST-v2-GateFourier | ok | accuracy | 0.514563 | 3.1182 | 0.0103 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v2_gate_fourier |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v4-CLS | synthetic_xor | 42 | classification | ablation | TST-v4-CLS | ok | accuracy | 0.514563 | 3.0189 | 0.0097 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v4_cls |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v4-Attention | synthetic_xor | 42 | classification | ablation | TST-v4-Attention | ok | accuracy | 0.514563 | 2.6481 | 0.0093 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v4_attention |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v5-CLS-Cross | synthetic_xor | 42 | classification | ablation | TST-v5-CLS-Cross | ok | accuracy | 0.495146 | 23.8205 | 0.0654 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v5_cls_cross |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | synthetic_irrelevant_noise | 42 | classification | baseline | mlp | ok | accuracy | 0.708738 | 0.0479 | 0.0019 | 512 | 100 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_irrelevant_noise | 42 | classification | baseline | lightgbm | ok | accuracy | 0.932039 | 0.0568 | 0.0027 | 512 | 100 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.504854 | 14.6121 | 0.0538 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v2-GateFourier | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v2-GateFourier | ok | accuracy | 0.504854 | 17.8335 | 0.0663 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v2_gate_fourier |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v4-CLS | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v4-CLS | ok | accuracy | 0.504854 | 31.3935 | 0.0805 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v4_cls |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v4-Attention | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v4-Attention | ok | accuracy | 0.504854 | 31.6735 | 0.2365 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v4_attention |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v5-CLS-Cross | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v5-CLS-Cross | ok | accuracy | 0.504854 | 114.3194 | 0.2058 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v5_cls_cross |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |

## Rank And Win Summary

- Total rows: 14
- Successful rows: 14
- Error rows: 0
- Lower mean rank is better; `status=error` rows are excluded from ranks and wins.

| Model | Family | Variant | Mean Rank | Wins | Ok Rows |
| --- | --- | --- | --- | --- | --- |
| LightGBM | baseline | lightgbm | 1.000 | 2 | 2 |
| MLP | baseline | mlp | 2.000 | 0 | 2 |
| TST-v1-Gate | ablation | TST-v1-Gate | 3.000 | 0 | 2 |
| TST-v2-GateFourier | ablation | TST-v2-GateFourier | 3.000 | 0 | 2 |
| TST-v4-Attention | ablation | TST-v4-Attention | 3.000 | 0 | 2 |
| TST-v4-CLS | ablation | TST-v4-CLS | 3.000 | 0 | 2 |
| TST-v5-CLS-Cross | ablation | TST-v5-CLS-Cross | 5.000 | 0 | 2 |

## Tree Vs TST Interpretation

- Tree mean rank: 1.000
- TST mean rank: 3.400
- Tree baselines remain ahead on this refresh. That is the empirical baseline for the next diagnostic pass, not a result to hide.

## Architecture Probe Verdict

- Research gate: failed.
- New variants did not improve the primary metric over the refreshed current-best TST on the same
  datasets, seed, sample size, and training budget.
- On `synthetic_xor`, TST-v4-CLS and TST-v4-Attention tied TST-v1/TST-v2 at 0.515 accuracy, while
  TST-v5-CLS-Cross scored 0.495.
- On `synthetic_irrelevant_noise`, all tested TST variants scored 0.505 accuracy.
- MLP and LightGBM remain clearly ahead, so these architecture-probe variants should not be scaled
  to a full fair comparison yet.
