| Model | Dataset | Metric | Score | Notes |
| --- | --- | --- | --- | --- |
| Linear/Ridge | synthetic_xor | accuracy | 0.5 | baseline |
| Random Forest | synthetic_xor | accuracy | 0.576923 | baseline |
| MLP | synthetic_xor | accuracy | 0.5 | baseline |
| TST-v0 | synthetic_xor | accuracy | 0.461538 | ablation |
| TST-v1-Gate | synthetic_xor | accuracy | 0.461538 | ablation |
| TST-v2-GateFourier | synthetic_xor | accuracy | 0.538462 | ablation |
| Linear/Ridge | synthetic_piecewise | rmse | 1.020553 | baseline |
| Random Forest | synthetic_piecewise | rmse | 0.746559 | baseline |
| MLP | synthetic_piecewise | rmse | 1.094903 | baseline |
| TST-v0 | synthetic_piecewise | rmse | 2.029315 | ablation |
| TST-v1-Gate | synthetic_piecewise | rmse | 2.034633 | ablation |
| TST-v2-GateFourier | synthetic_piecewise | rmse | 2.885752 | ablation |
