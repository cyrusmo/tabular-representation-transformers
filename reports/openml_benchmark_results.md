| Model | Dataset | Seed | Task | Family | Variant | Status | Metric | Score | Fit Seconds | Predict Seconds | N Samples | N Features | Artifact Path | Error Message | Notes | Benchmark Mode | Base Variant | Selected Config Id | Selected Lr | Selection Metric | Selection Mode | Selection Score | Selected Epoch | Candidate Config Count | Candidate Lrs | Tuning Budget Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear/Ridge | adult | 42 | classification | baseline | linear | ok | accuracy | 0.805825 | 0.0271 | 0.0054 | 512 | 14 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | adult | 42 | classification | baseline | mlp | ok | accuracy | 0.84466 | 0.1148 | 0.0049 | 512 | 14 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | adult | 42 | classification | baseline | lightgbm | ok | accuracy | 0.796117 | 3.4175 | 0.0095 | 512 | 14 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | adult | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.757282 | 9.2628 | 0.061 | 512 | 14 | outputs/benchmark_artifacts/adult/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | adult | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.757282 | 7.9592 | 0.0517 | 512 | 14 | outputs/benchmark_artifacts/adult/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | bank-marketing | 42 | classification | baseline | linear | ok | accuracy | 0.92233 | 0.0132 | 0.0038 | 512 | 16 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | bank-marketing | 42 | classification | baseline | mlp | ok | accuracy | 0.932039 | 0.0468 | 0.0056 | 512 | 16 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | bank-marketing | 42 | classification | baseline | lightgbm | ok | accuracy | 0.883495 | 0.0173 | 0.0055 | 512 | 16 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | bank-marketing | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.873786 | 5.1754 | 0.0507 | 512 | 16 | outputs/benchmark_artifacts/bank-marketing/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | bank-marketing | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.873786 | 5.0194 | 0.0424 | 512 | 16 | outputs/benchmark_artifacts/bank-marketing/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | covertype | 42 | classification | baseline | linear | ok | accuracy | 0.61165 | 0.0895 | 0.0189 | 512 | 54 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | covertype | 42 | classification | baseline | mlp | ok | accuracy | 0.621359 | 0.1438 | 0.0212 | 512 | 54 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | covertype | 42 | classification | baseline | lightgbm | ok | accuracy | 0.621359 | 0.209 | 0.0425 | 512 | 54 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | covertype | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.485437 | 8.62 | 0.0518 | 512 | 54 | outputs/benchmark_artifacts/covertype/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | covertype | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.485437 | 8.1096 | 0.0687 | 512 | 54 | outputs/benchmark_artifacts/covertype/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | higgs-small | 42 | classification | baseline | linear | ok | accuracy | 0.592233 | 0.0111 | 0.002 | 512 | 28 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | higgs-small | 42 | classification | baseline | mlp | ok | accuracy | 0.572816 | 0.0474 | 0.0021 | 512 | 28 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | higgs-small | 42 | classification | baseline | lightgbm | ok | accuracy | 0.601942 | 0.0241 | 0.0058 | 512 | 28 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | higgs-small | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.543689 | 2.6601 | 0.0265 | 512 | 28 | outputs/benchmark_artifacts/higgs-small/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | higgs-small | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.543689 | 2.5567 | 0.0314 | 512 | 28 | outputs/benchmark_artifacts/higgs-small/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | heloc | 42 | classification | baseline | linear | ok | accuracy | 0.669903 | 0.0054 | 0.0012 | 512 | 7 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | heloc | 42 | classification | baseline | mlp | ok | accuracy | 0.747573 | 0.0338 | 0.0019 | 512 | 7 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | heloc | 42 | classification | baseline | lightgbm | ok | accuracy | 0.786408 | 0.0128 | 0.0035 | 512 | 7 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | heloc | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.514563 | 0.6698 | 0.0051 | 512 | 7 | outputs/benchmark_artifacts/heloc/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | heloc | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.514563 | 0.6897 | 0.0059 | 512 | 7 | outputs/benchmark_artifacts/heloc/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | california-housing | 42 | regression | baseline | linear | ok | rmse | 67873.691152 | 0.0475 | 0.0043 | 512 | 9 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | california-housing | 42 | regression | baseline | mlp | ok | rmse | 237034.027305 | 0.0349 | 0.0036 | 512 | 9 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | california-housing | 42 | regression | baseline | lightgbm | ok | rmse | 61483.351142 | 0.0165 | 0.0034 | 512 | 9 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | california-housing | 42 | regression | ablation | TST-v0 | ok | rmse | 237040.492902 | 1.3755 | 0.0091 | 512 | 9 | outputs/benchmark_artifacts/california-housing/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | california-housing | 42 | regression | ablation | TST-v1-Gate | ok | rmse | 237040.490555 | 1.2244 | 0.0131 | 512 | 9 | outputs/benchmark_artifacts/california-housing/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | credit-g | 42 | classification | baseline | linear | ok | accuracy | 0.679612 | 0.0135 | 0.0035 | 512 | 20 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | credit-g | 42 | classification | baseline | mlp | ok | accuracy | 0.640777 | 0.054 | 0.0073 | 512 | 20 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | credit-g | 42 | classification | baseline | lightgbm | ok | accuracy | 0.699029 | 0.0199 | 0.0061 | 512 | 20 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | credit-g | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.708738 | 5.9595 | 0.0471 | 512 | 20 | outputs/benchmark_artifacts/credit-g/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | credit-g | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.708738 | 5.9687 | 0.0503 | 512 | 20 | outputs/benchmark_artifacts/credit-g/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| Linear/Ridge | jannis | 42 | classification | baseline | linear | ok | accuracy | 0.660194 | 0.0358 | 0.0019 | 512 | 54 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| MLP | jannis | 42 | classification | baseline | mlp | ok | accuracy | 0.631068 | 0.0745 | 0.0057 | 512 | 54 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | jannis | 42 | classification | baseline | lightgbm | ok | accuracy | 0.640777 | 0.2213 | 0.0059 | 512 | 54 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | jannis | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.543689 | 6.823 | 0.0388 | 512 | 54 | outputs/benchmark_artifacts/jannis/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | jannis | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.456311 | 9.2594 | 0.1403 | 512 | 54 | outputs/benchmark_artifacts/jannis/seed_42/tst_v1_gate |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |

## Rank And Win Summary

- Total rows: 40
- Successful rows: 40
- Error rows: 0
- Lower mean rank is better; `status=error` rows are excluded from ranks and wins.

| Model | Family | Variant | Mean Rank | Wins | Ok Rows |
| --- | --- | --- | --- | --- | --- |
| LightGBM | baseline | lightgbm | 1.875 | 4 | 8 |
| Linear/Ridge | baseline | linear | 2.375 | 1 | 8 |
| MLP | baseline | mlp | 2.375 | 3 | 8 |
| TST-v0 | ablation | TST-v0 | 3.750 | 1 | 8 |
| TST-v1-Gate | ablation | TST-v1-Gate | 3.750 | 1 | 8 |

## Tree Vs TST Interpretation

- Tree mean rank: 1.875
- TST mean rank: 3.750
- Tree baselines remain ahead on this refresh. That is the empirical baseline for the next diagnostic pass, not a result to hide.

