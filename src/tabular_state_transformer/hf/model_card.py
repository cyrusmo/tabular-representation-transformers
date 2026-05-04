from __future__ import annotations


def make_model_card(model_name: str = "Tabular State Transformer") -> str:
    return f"""---
library_name: pytorch
license: mit
tags:
  - tabular-classification
  - tabular-regression
  - experimental
  - pytorch
---

# {model_name}

## Model Summary

Tabular State Transformer is an experimental PyTorch toolkit for tabular prediction tasks where feature identity, sparse interactions, non-smooth boundaries, and regime-like behavior matter.

The primary contribution is the ablation harness, not a claim of universal SOTA performance.

## Intended Use

Research, benchmarking, and ablation studies on synthetic and public tabular datasets.

## Out-of-Scope Use

This model should not be used as investment advice, a trading model, or a universal replacement for tree-based baselines.

## Architecture

The default TST-v0 model uses feature tokenization, interaction blocks, and a simple prediction head. Sparse gates, Fourier features, and mixture-of-experts heads are opt-in ablations.

## Training Data

Public v1 is designed around synthetic stress tests and a small curated OpenML suite.

## Evaluation

Evaluate against linear/ridge, random forest, MLP, and optional gradient-boosted tree baselines before making claims.

## Limitations

The toolkit is experimental. It may underperform boosted trees, can overfit high-frequency expansions, and currently supports categorical features through sklearn preprocessing rather than native embeddings.

## Ethical / Misuse Considerations

Do not present benchmark results as investment advice, credit decisions, medical recommendations, or a claim of broad deployment readiness.

## Citation

If you use this experimental toolkit, cite the repository.

## How to Use

```python
from tabular_state_transformer import TabularStateTransformer

model = TabularStateTransformer.from_pretrained(\"./exported_model\")
preds = model.predict_numpy(X_preprocessed)
```
"""
