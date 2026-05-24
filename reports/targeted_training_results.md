| Model | Dataset | Seed | Task | Family | Variant | Status | Metric | Score | Fit Seconds | Predict Seconds | N Samples | N Features | Artifact Path | Error Message | Notes | Benchmark Mode | Base Variant | Selected Config Id | Selected Lr | Selection Metric | Selection Mode | Selection Score | Selected Epoch | Candidate Config Count | Candidate Lrs | Tuning Budget Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | synthetic_xor | 42 | classification | baseline | mlp | ok | accuracy | 0.679612 | 0.0427 | 0.0021 | 512 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_xor | 42 | classification | baseline | lightgbm | ok | accuracy | 0.990291 | 0.2408 | 0.0042 | 512 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_xor | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.514563 | 6.1028 | 0.0249 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v2-GateFourier | synthetic_xor | 42 | classification | ablation | TST-v2-GateFourier | ok | accuracy | 0.514563 | 4.6038 | 0.0281 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v2_gate_fourier |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_xor | 42 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.533981 | 14.6475 | 0.0249 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.543689 | 4 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 42 | classification | ablation | TST-v2-GateFourier-tuned | ok | accuracy | 0.514563 | 14.8312 | 0.0156 | 512 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v2_gate_fourier_tuned |  |  | tuned_tst_benchmark | TST-v2-GateFourier | tst_v2_gate_fourier_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.514563 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| MLP | synthetic_irrelevant_noise | 42 | classification | baseline | mlp | ok | accuracy | 0.708738 | 0.0524 | 0.0026 | 512 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_irrelevant_noise | 42 | classification | baseline | lightgbm | ok | accuracy | 0.932039 | 0.0971 | 0.0045 | 512 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.504854 | 25.3918 | 0.0971 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v2-GateFourier | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v2-GateFourier | ok | accuracy | 0.504854 | 28.1423 | 0.1033 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v2_gate_fourier |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.485437 | 72.1809 | 0.0745 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.514563 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v2-GateFourier-tuned | ok | accuracy | 0.504854 | 83.9903 | 0.0856 | 512 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v2_gate_fourier_tuned |  |  | tuned_tst_benchmark | TST-v2-GateFourier | tst_v2_gate_fourier_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.524272 | 4 | 3 | 1e-4,3e-4,1e-3 | lr_only |

## Rank And Win Summary

- Total rows: 12
- Successful rows: 12
- Error rows: 0
- Lower mean rank is better; `status=error` rows are excluded from ranks and wins.

| Model | Family | Variant | Mean Rank | Wins | Ok Rows |
| --- | --- | --- | --- | --- | --- |
| LightGBM | baseline | lightgbm | 1.000 | 2 | 2 |
| MLP | baseline | mlp | 2.000 | 0 | 2 |
| TST | ablation | TST-v2-GateFourier-tuned | 3.500 | 0 | 2 |
| TST-v1-Gate | ablation | TST-v1-Gate | 3.500 | 0 | 2 |
| TST-v2-GateFourier | ablation | TST-v2-GateFourier | 3.500 | 0 | 2 |
| TST | ablation | TST-v1-Gate-tuned | 4.500 | 0 | 2 |

## Tree Vs TST Interpretation

- Tree mean rank: 1.000
- TST mean rank: 3.500
- Tree baselines remain ahead on this refresh. That is the empirical baseline for the next diagnostic pass, not a result to hide.

## Fair TST Tuning Summary

- Untuned TST mean rank: 3.500
- Tuned TST mean rank: 4.000
- Tuning improvement: -0.500 mean-rank points
- Remaining gap to trees: 3.000 mean-rank points
- Tuned rows use a fixed three-candidate learning-rate budget, not a broad hyperparameter search.

