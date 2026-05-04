from __future__ import annotations

import torch
from tabular_state_transformer import TabularStateConfig, TabularStateTransformer

def test_model_forward_regression_shape():
    model = TabularStateTransformer(TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1))
    y = model(torch.randn(8, 6))
    assert y.shape == (8,)
