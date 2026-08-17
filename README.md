---
library_name: pytorch
license: mit
tags:
  - tabular-classification
  - tabular-regression
  - experimental
  - pytorch
---

# Tabular State Transformer

Tabular State Transformer is an experimental PyTorch toolkit for tabular prediction tasks where feature identity, sparse interactions, non-smooth boundaries, and regime-like behavior matter.

This project tests whether preserving feature identity, adding sparse feature gating, and using spectral/interaction blocks can improve neural tabular robustness against strong baselines.

The primary contribution is the ablation harness, not a claim of universal SOTA performance.

## Intended Use

Research, benchmarking, and ablation studies on tabular datasets. This is not a production financial trading system and does not claim to produce tradable alpha.

## Out-of-Scope Use

Do not use this repository as investment advice, a trading model, or a foundation model claim.

## Quickstart

```python
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.training import Trainer

bundle = load_dataset("synthetic_xor", split_seed=42)
config = TabularStateConfig(n_features=20, task="classification", max_epochs=2)
result = Trainer(config).fit(bundle)
```

```bash
pip install -e ".[dev,benchmark]"
python scripts/make_synthetic.py --config configs/data/synthetic.yaml
python scripts/train.py --config configs/experiment/classification.yaml
python scripts/run_benchmark.py --suite synthetic
```

## Architecture

```text
Raw tabular features
  -> FeatureTokenizer
  -> optional SparseFeatureGate
  -> optional FourierFeatureBlock
  -> InteractionBlock
  -> ClassificationHead or RegressionHead
```

The default configuration is **TST-v0**: no gate, no Fourier block, no MoE, and a simple prediction head. Explicit ablation configs are provided:

- **TST-v0:** `configs/model/tst_v0.yaml`
- **TST-v1-Gate:** `configs/model/tst_v1_gate.yaml`
- **TST-v2-GateFourier:** `configs/model/tst_v2_fourier_gate.yaml`
- **TST-v3-MoE:** `configs/model/tst_v3_moe.yaml`

## Benchmark Plan

Compare honestly against linear/ridge models, MLP, random forest, optional LightGBM/XGBoost, and the TST ablations.

Public v1 supports DataFrames with categorical columns via sklearn preprocessing, not native categorical embeddings. The model receives a preprocessed `float32` matrix.

## Limitations

This toolkit is experimental. Under the published synthetic-stress protocol, TST ablations do **not**
beat strong tree baselines on interaction-heavy or high-dimensional noise tasks (notably XOR). The
intended contribution is a reproducible ablation/diagnostic harness and an honest negative result
(`reports/analysis/research_narrative.md`), not a claim of universal SOTA performance. The model may
overfit high-frequency expansions and has not been validated for production or trading use
(`docs/finance_disclaimer.md`).
