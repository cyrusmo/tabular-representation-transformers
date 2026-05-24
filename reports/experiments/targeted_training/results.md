| Model | Dataset | Seed | Task | Family | Variant | Status | Metric | Score | Fit Seconds | Predict Seconds | N Samples | N Features | Artifact Path | Error Message | Notes | Benchmark Mode | Base Variant | Selected Config Id | Selected Lr | Selection Metric | Selection Mode | Selection Score | Selected Epoch | Candidate Config Count | Candidate Lrs | Tuning Budget Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | synthetic_xor | 42 | classification | baseline | mlp | ok | accuracy | 0.765854 | 0.089 | 0.0028 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_xor | 42 | classification | baseline | lightgbm | ok | accuracy | 0.985366 | 0.2769 | 0.0067 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| Gradient Boosting | synthetic_xor | 42 | classification | baseline | gradient_boosting | ok | accuracy | 0.970732 | 0.3341 | 0.0024 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | synthetic_xor | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.512195 | 13.1503 | 0.0569 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v0 |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_xor | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.512195 | 13.877 | 0.0617 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v3-MoE | synthetic_xor | 42 | classification | ablation | TST-v3-MoE | ok | accuracy | 0.512195 | 21.9102 | 0.0991 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v3_moe |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| FT-Transformer-style | synthetic_xor | 42 | classification | neural_baseline | local_ft_transformer | ok | accuracy | 0.395122 | 13.3484 | 0.0443 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_xor | 42 | classification | ablation | TST-v0-tuned | ok | accuracy | 0.492683 | 90.3812 | 0.0665 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v0_tuned |  |  | tuned_tst_benchmark | TST-v0 | tst_v0_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.560976 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 42 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.487805 | 49.4993 | 0.0413 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.570732 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 42 | classification | ablation | TST-v3-MoE-tuned | ok | accuracy | 0.512195 | 36.8758 | 0.055 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_42/tst_v3_moe_tuned |  |  | tuned_tst_benchmark | TST-v3-MoE | tst_v3_moe_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.512195 | 2 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| MLP | synthetic_irrelevant_noise | 42 | classification | baseline | mlp | ok | accuracy | 0.795122 | 0.159 | 0.0025 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_irrelevant_noise | 42 | classification | baseline | lightgbm | ok | accuracy | 0.917073 | 0.7847 | 0.0085 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| Gradient Boosting | synthetic_irrelevant_noise | 42 | classification | baseline | gradient_boosting | ok | accuracy | 0.907317 | 1.5426 | 0.013 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.502439 | 73.6491 | 0.2496 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v0 |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.502439 | 78.2355 | 0.2972 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v3-MoE | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v3-MoE | ok | accuracy | 0.502439 | 97.5405 | 0.3734 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v3_moe |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| FT-Transformer-style | synthetic_irrelevant_noise | 42 | classification | neural_baseline | local_ft_transformer | ok | accuracy | 0.926829 | 124.0448 | 0.1046 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v0-tuned | ok | accuracy | 0.84878 | 890.7557 | 0.2271 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v0_tuned |  |  | tuned_tst_benchmark | TST-v0 | tst_v0_lr_3e-4 | 3e-4 | val_accuracy | maximize | 0.882927 | 48 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.829268 | 410.8118 | 0.2781 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.887805 | 46 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 42 | classification | ablation | TST-v3-MoE-tuned | ok | accuracy | 0.502439 | 199.1712 | 0.2445 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_42/tst_v3_moe_tuned |  |  | tuned_tst_benchmark | TST-v3-MoE | tst_v3_moe_lr_3e-4 | 3e-4 | val_accuracy | maximize | 0.502439 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| MLP | synthetic_xor | 43 | classification | baseline | mlp | ok | accuracy | 0.8 | 0.148 | 0.0033 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_xor | 43 | classification | baseline | lightgbm | ok | accuracy | 1.0 | 0.1167 | 0.0081 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| Gradient Boosting | synthetic_xor | 43 | classification | baseline | gradient_boosting | ok | accuracy | 0.795122 | 0.4118 | 0.0056 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | synthetic_xor | 43 | classification | ablation | TST-v0 | ok | accuracy | 0.526829 | 12.5889 | 0.0437 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_43/tst_v0 |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_xor | 43 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.526829 | 11.4651 | 0.0483 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_43/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v3-MoE | synthetic_xor | 43 | classification | ablation | TST-v3-MoE | ok | accuracy | 0.526829 | 11.8383 | 0.0385 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_43/tst_v3_moe |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| FT-Transformer-style | synthetic_xor | 43 | classification | neural_baseline | local_ft_transformer | ok | accuracy | 0.526829 | 4.7559 | 0.0182 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_xor | 43 | classification | ablation | TST-v0-tuned | ok | accuracy | 0.507317 | 40.8966 | 0.0611 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_43/tst_v0_tuned |  |  | tuned_tst_benchmark | TST-v0 | tst_v0_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.560976 | 3 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 43 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.492683 | 37.6905 | 0.0386 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_43/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.536585 | 2 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 43 | classification | ablation | TST-v3-MoE-tuned | ok | accuracy | 0.526829 | 38.0306 | 0.0486 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_43/tst_v3_moe_tuned |  |  | tuned_tst_benchmark | TST-v3-MoE | tst_v3_moe_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.526829 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| MLP | synthetic_irrelevant_noise | 43 | classification | baseline | mlp | ok | accuracy | 0.765854 | 0.1283 | 0.0027 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_irrelevant_noise | 43 | classification | baseline | lightgbm | ok | accuracy | 0.941463 | 0.4221 | 0.0073 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| Gradient Boosting | synthetic_irrelevant_noise | 43 | classification | baseline | gradient_boosting | ok | accuracy | 0.931707 | 1.5055 | 0.0041 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | synthetic_irrelevant_noise | 43 | classification | ablation | TST-v0 | ok | accuracy | 0.512195 | 68.023 | 0.1822 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_43/tst_v0 |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_irrelevant_noise | 43 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.512195 | 64.5821 | 0.1977 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_43/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v3-MoE | synthetic_irrelevant_noise | 43 | classification | ablation | TST-v3-MoE | ok | accuracy | 0.512195 | 55.9781 | 0.2286 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_43/tst_v3_moe |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| FT-Transformer-style | synthetic_irrelevant_noise | 43 | classification | neural_baseline | local_ft_transformer | ok | accuracy | 0.907317 | 59.1565 | 0.0712 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_irrelevant_noise | 43 | classification | ablation | TST-v0-tuned | ok | accuracy | 0.809756 | 462.313 | 0.3108 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_43/tst_v0_tuned |  |  | tuned_tst_benchmark | TST-v0 | tst_v0_lr_3e-4 | 3e-4 | val_accuracy | maximize | 0.878049 | 45 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 43 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.819512 | 281.8438 | 0.1814 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_43/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.897561 | 42 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 43 | classification | ablation | TST-v3-MoE-tuned | ok | accuracy | 0.512195 | 125.7178 | 0.2008 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_43/tst_v3_moe_tuned |  |  | tuned_tst_benchmark | TST-v3-MoE | tst_v3_moe_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.512195 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| MLP | synthetic_xor | 44 | classification | baseline | mlp | ok | accuracy | 0.756098 | 0.2735 | 0.0039 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_xor | 44 | classification | baseline | lightgbm | ok | accuracy | 0.995122 | 0.0908 | 0.0072 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| Gradient Boosting | synthetic_xor | 44 | classification | baseline | gradient_boosting | ok | accuracy | 0.82439 | 0.2886 | 0.0023 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | synthetic_xor | 44 | classification | ablation | TST-v0 | ok | accuracy | 0.526829 | 8.5065 | 0.0283 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_44/tst_v0 |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_xor | 44 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.526829 | 7.935 | 0.0289 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_44/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v3-MoE | synthetic_xor | 44 | classification | ablation | TST-v3-MoE | ok | accuracy | 0.526829 | 8.4496 | 0.0398 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_44/tst_v3_moe |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| FT-Transformer-style | synthetic_xor | 44 | classification | neural_baseline | local_ft_transformer | ok | accuracy | 0.526829 | 2.7835 | 0.0127 | 1024 | 20 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_xor | 44 | classification | ablation | TST-v0-tuned | ok | accuracy | 0.531707 | 40.7258 | 0.0287 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_44/tst_v0_tuned |  |  | tuned_tst_benchmark | TST-v0 | tst_v0_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.536585 | 8 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 44 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.517073 | 29.2972 | 0.039 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_44/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.536585 | 9 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_xor | 44 | classification | ablation | TST-v3-MoE-tuned | ok | accuracy | 0.526829 | 26.7344 | 0.0281 | 1024 | 20 | outputs/benchmark_artifacts/synthetic_xor/seed_44/tst_v3_moe_tuned |  |  | tuned_tst_benchmark | TST-v3-MoE | tst_v3_moe_lr_3e-4 | 3e-4 | val_accuracy | maximize | 0.526829 | 1 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| MLP | synthetic_irrelevant_noise | 44 | classification | baseline | mlp | ok | accuracy | 0.858537 | 0.0831 | 0.0021 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| LightGBM | synthetic_irrelevant_noise | 44 | classification | baseline | lightgbm | ok | accuracy | 0.941463 | 0.616 | 0.0112 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| Gradient Boosting | synthetic_irrelevant_noise | 44 | classification | baseline | gradient_boosting | ok | accuracy | 0.941463 | 1.0585 | 0.0026 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | synthetic_irrelevant_noise | 44 | classification | ablation | TST-v0 | ok | accuracy | 0.517073 | 36.6933 | 0.1464 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_44/tst_v0 |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v1-Gate | synthetic_irrelevant_noise | 44 | classification | ablation | TST-v1-Gate | ok | accuracy | 0.517073 | 42.5456 | 0.1473 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_44/tst_v1_gate |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v3-MoE | synthetic_irrelevant_noise | 44 | classification | ablation | TST-v3-MoE | ok | accuracy | 0.517073 | 40.6431 | 0.1466 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_44/tst_v3_moe |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| FT-Transformer-style | synthetic_irrelevant_noise | 44 | classification | neural_baseline | local_ft_transformer | ok | accuracy | 0.912195 | 57.8236 | 0.0559 | 1024 | 100 |  |  |  | tuned_tst_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST | synthetic_irrelevant_noise | 44 | classification | ablation | TST-v0-tuned | ok | accuracy | 0.702439 | 247.7153 | 0.1551 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_44/tst_v0_tuned |  |  | tuned_tst_benchmark | TST-v0 | tst_v0_lr_3e-4 | 3e-4 | val_accuracy | maximize | 0.643902 | 16 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 44 | classification | ablation | TST-v1-Gate-tuned | ok | accuracy | 0.887805 | 294.8127 | 0.1636 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_44/tst_v1_gate_tuned |  |  | tuned_tst_benchmark | TST-v1-Gate | tst_v1_gate_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.853659 | 50 | 3 | 1e-4,3e-4,1e-3 | lr_only |
| TST | synthetic_irrelevant_noise | 44 | classification | ablation | TST-v3-MoE-tuned | ok | accuracy | 0.517073 | 157.2312 | 0.2264 | 1024 | 100 | outputs/benchmark_artifacts/synthetic_irrelevant_noise/seed_44/tst_v3_moe_tuned |  |  | tuned_tst_benchmark | TST-v3-MoE | tst_v3_moe_lr_1e-4 | 1e-4 | val_accuracy | maximize | 0.521951 | 6 | 3 | 1e-4,3e-4,1e-3 | lr_only |

## Rank And Win Summary

- Total rows: 60
- Successful rows: 60
- Error rows: 0
- Lower mean rank is better; `status=error` rows are excluded from ranks and wins.

| Model | Family | Variant | Mean Rank | Wins | Ok Rows |
| --- | --- | --- | --- | --- | --- |
| LightGBM | baseline | lightgbm | 1.167 | 5 | 6 |
| Gradient Boosting | baseline | gradient_boosting | 2.167 | 1 | 6 |
| MLP | baseline | mlp | 4.167 | 0 | 6 |
| FT-Transformer-style | neural_baseline | local_ft_transformer | 4.333 | 1 | 6 |
| TST | ablation | TST-v3-MoE-tuned | 5.667 | 0 | 6 |
| TST-v0 | ablation | TST-v0 | 5.667 | 0 | 6 |
| TST-v1-Gate | ablation | TST-v1-Gate | 5.667 | 0 | 6 |
| TST-v3-MoE | ablation | TST-v3-MoE | 5.667 | 0 | 6 |
| TST | ablation | TST-v0-tuned | 6.000 | 0 | 6 |
| TST | ablation | TST-v1-Gate-tuned | 7.000 | 0 | 6 |

## Tree Vs TST Interpretation

- Tree mean rank: 1.667
- TST mean rank: 5.667
- Tree baselines remain ahead on this refresh. That is the empirical baseline for the next diagnostic pass, not a result to hide.

## Fair TST Tuning Summary

- Untuned TST mean rank: 5.667
- Tuned TST mean rank: 6.222
- Tuning improvement: -0.556 mean-rank points
- Remaining gap to trees: 4.556 mean-rank points
- Tuned rows use a fixed three-candidate learning-rate budget, not a broad hyperparameter search.

## Neural Baseline Summary

- Local FT-Transformer-style mean rank: 4.333
- This is a local FT-Transformer-style baseline, not a validated reference-paper reproduction.

