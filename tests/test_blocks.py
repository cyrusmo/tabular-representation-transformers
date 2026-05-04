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
