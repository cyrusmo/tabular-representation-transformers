from __future__ import annotations

import torch
from tabular_state_transformer.blocks.gate import SparseFeatureGate
from tabular_state_transformer.blocks.spectral import FourierFeatureBlock, WaveletFeatureBlock

def test_blocks_preserve_token_shape():
    x = torch.randn(4, 5, 16)
    assert SparseFeatureGate(5)(x).shape == x.shape
    assert FourierFeatureBlock(16)(x).shape == x.shape
    assert WaveletFeatureBlock(16)(x).shape == x.shape
