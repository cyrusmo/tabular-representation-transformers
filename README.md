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

An experimental neural architecture/toolkit for non-smooth, interaction-heavy tabular tasks, motivated by the known weaknesses of generic neural models on tabular data.

This project explores whether tabular models can benefit from preserving feature identity, adding sparse feature gates, exposing spectral/wavelet structure, learning feature interactions, and using regime-gated heads.

## Intended Use

Research, benchmarking, and ablation studies on tabular datasets. This is not a production financial trading system and does not claim to produce tradable alpha.

## Out-of-Scope Use

Do not use this repository as investment advice, a trading model, or a foundation model claim.

## Quickstart

```python
from tabular_state_transformer.sklearn_api import TabularStateRegressor
from tabular_state_transformer.data.synthetic import make_threshold_regression

X, y = make_threshold_regression(n_samples=2000, n_features=20, random_state=42)
model = TabularStateRegressor(max_epochs=5, d_token=32)
model.fit(X, y)
print(model.predict(X[:5]))
```

## Architecture

```text
Raw tabular features
  -> FeatureTokenizer
  -> SparseFeatureGate
  -> Fourier/Wavelet Expansion
  -> InteractionBlock
  -> RegimeGatedHead
  -> Classification or Regression head
```

## Benchmark Plan

Compare honestly against MLP, random forest, LightGBM/XGBoost, FT-Transformer/TabTransformer, TabPFN where appropriate, and simple linear baselines.

## Limitations

This toolkit is experimental. It may underperform boosted trees, can overfit high-frequency expansions, and has not been validated for production use.
