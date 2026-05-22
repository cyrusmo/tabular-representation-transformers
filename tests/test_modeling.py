from __future__ import annotations

import torch
from tabular_state_transformer import TabularStateConfig, TabularStateTransformer
from tabular_state_transformer.models import FTTransformerStyle

def test_model_forward_regression_shape():
    model = TabularStateTransformer(TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1))
    y = model(torch.randn(8, 6))
    assert y.shape == (8,)


def test_model_forward_classification_shape():
    model = TabularStateTransformer(
        TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1, task="classification", n_classes=3)
    )
    y = model(torch.randn(8, 6))
    assert y.shape == (8, 3)


def test_default_model_uses_simple_head():
    model = TabularStateTransformer(TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1))
    assert model.config.use_gate is False
    assert model.config.use_fourier is False
    assert model.config.use_moe is False


def test_ft_transformer_style_forward_regression_shape():
    model = FTTransformerStyle(TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1))
    y = model(torch.randn(8, 6))
    assert y.shape == (8,)


def test_ft_transformer_style_forward_classification_shape():
    model = FTTransformerStyle(
        TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1, task="classification", n_classes=3)
    )
    y = model(torch.randn(8, 6))
    assert y.shape == (8, 3)


def test_ft_transformer_style_predict_numpy_classification_probabilities():
    model = FTTransformerStyle(
        TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1, task="classification", n_classes=3)
    )
    y = model.predict_numpy(torch.randn(8, 6).numpy())
    assert y.shape == (8, 3)
    assert torch.allclose(torch.as_tensor(y).sum(dim=1), torch.ones(8), atol=1e-6)
