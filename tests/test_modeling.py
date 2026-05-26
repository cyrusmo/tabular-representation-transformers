from __future__ import annotations

import pytest
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
    y = model(torch.randn(8, 6))
    assert y.shape == (8,)


def test_tst_cls_pooling_forward_shapes():
    regression_model = TabularStateTransformer(
        TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1, pooling="cls")
    )
    classification_model = TabularStateTransformer(
        TabularStateConfig(
            n_features=6,
            d_token=16,
            n_heads=4,
            n_layers=1,
            pooling="cls",
            task="classification",
            n_classes=3,
        )
    )
    assert regression_model(torch.randn(8, 6)).shape == (8,)
    assert classification_model(torch.randn(8, 6)).shape == (8, 3)


def test_tst_attention_pooling_forward_shapes():
    regression_model = TabularStateTransformer(
        TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1, pooling="attention")
    )
    classification_model = TabularStateTransformer(
        TabularStateConfig(
            n_features=6,
            d_token=16,
            n_heads=4,
            n_layers=1,
            pooling="attention",
            task="classification",
            n_classes=3,
        )
    )
    assert regression_model(torch.randn(8, 6)).shape == (8,)
    assert classification_model(torch.randn(8, 6)).shape == (8, 3)


def test_feature_cross_forward_pass_for_20_and_100_features():
    for n_features in (20, 100):
        model = TabularStateTransformer(
            TabularStateConfig(
                n_features=n_features,
                d_token=16,
                n_heads=4,
                n_layers=1,
                use_feature_crosses=True,
                cross_max_features=16,
            )
        )
        y = model(torch.randn(4, n_features))
        assert y.shape == (4,)


def test_feature_cross_token_count_respects_cross_max_features():
    model_20 = TabularStateTransformer(
        TabularStateConfig(
            n_features=20,
            d_token=16,
            n_heads=4,
            n_layers=1,
            use_feature_crosses=True,
            cross_max_features=16,
        )
    )
    tokens_20 = model_20._tokens_before_interaction(torch.randn(2, 20))
    assert model_20.cross_tokenizer is not None
    assert model_20.cross_tokenizer.num_cross_tokens == 120
    assert tokens_20.shape == (2, 140, 16)

    model_100 = TabularStateTransformer(
        TabularStateConfig(
            n_features=100,
            d_token=16,
            n_heads=4,
            n_layers=1,
            use_feature_crosses=True,
            cross_max_features=16,
        )
    )
    tokens_100 = model_100._tokens_before_interaction(torch.randn(2, 100))
    assert model_100.cross_tokenizer is not None
    assert model_100.cross_tokenizer.num_cross_tokens == 120
    assert tokens_100.shape == (2, 220, 16)


def test_invalid_pooling_raises():
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        TabularStateTransformer(
            TabularStateConfig(n_features=6, d_token=16, n_heads=4, n_layers=1, pooling="bogus")
        )


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
