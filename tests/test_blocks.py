from __future__ import annotations

import torch
from tabular_state_transformer.blocks.gate import SparseFeatureGate
from tabular_state_transformer.blocks.spectral import FourierFeatureBlock, WaveletFeatureBlock

def test_blocks_preserve_token_shape():
    x = torch.randn(4, 5, 16)
    assert SparseFeatureGate(5)(x).shape == x.shape
    assert FourierFeatureBlock(16)(x).shape == x.shape
    assert WaveletFeatureBlock(16)(x).shape == x.shape


def test_gate_values_are_bounded_and_finite():
    gate = SparseFeatureGate(5)
    values = torch.sigmoid(gate.logits)
    assert torch.isfinite(values).all()
    assert torch.all((values >= 0) & (values <= 1))


def test_gate_can_sparsify_under_strong_l1():
    from tabular_state_transformer.config import TabularStateConfig
    from tabular_state_transformer.data import load_dataset
    from tabular_state_transformer.training import Trainer
    from tabular_state_transformer.training.diagnostics import gate_summary

    bundle = load_dataset("synthetic_irrelevant_noise", split_seed=42, n_samples=256, n_features=50)
    config = TabularStateConfig(
        n_features=50,
        d_token=16,
        n_heads=2,
        n_layers=1,
        use_gate=True,
        gate_init=-2.0,
        gate_l1=0.01,
        gate_lr_multiplier=200.0,
        task="classification",
        max_epochs=15,
        batch_size=64,
        early_stopping_patience=None,
    )
    result = Trainer(config).fit(bundle)
    summary = gate_summary(result.model)
    assert summary["has_gate"] is True
    assert float(summary["gate_sparsity"]) > 0.0
