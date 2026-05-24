| Model | Dataset | Seed | Task | Family | Variant | Status | Metric | Score | Fit Seconds | Predict Seconds | N Samples | N Features | Artifact Path | Error Message | Notes | Benchmark Mode | Base Variant | Selected Config Id | Selected Lr | Selection Metric | Selection Mode | Selection Score | Selected Epoch | Candidate Config Count | Candidate Lrs | Tuning Budget Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear/Ridge | credit-g | 42 | classification | baseline | linear | ok | accuracy | 0.75 | 0.0173 | 0.0087 | 256 | 20 |  |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |
| TST-v0 | credit-g | 42 | classification | ablation | TST-v0 | ok | accuracy | 0.711538 | 1.7901 | 0.0373 | 256 | 20 | outputs/benchmark_artifacts/credit-g/seed_42/tst_v0 |  |  | default_benchmark |  |  |  |  |  |  |  |  |  |  |

## Rank And Win Summary

- Total rows: 2
- Successful rows: 2
- Error rows: 0
- Lower mean rank is better; `status=error` rows are excluded from ranks and wins.

| Model | Family | Variant | Mean Rank | Wins | Ok Rows |
| --- | --- | --- | --- | --- | --- |
| Linear/Ridge | baseline | linear | 1.000 | 1 | 1 |
| TST-v0 | ablation | TST-v0 | 2.000 | 0 | 1 |

## Tree Vs TST Interpretation

- Tree mean rank: unavailable
- TST mean rank: 2.000
- Tree-vs-TST comparison is unavailable because one side has no successful rows.

