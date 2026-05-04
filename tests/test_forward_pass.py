from __future__ import annotations

import torch

from tabular_state_transformer import TabularStateConfig, TabularStateTransformer


def test_tst_v1_and_v2_forward_passes():
    x = torch.randn(4, 8)
    gate_model = TabularStateTransformer(
        TabularStateConfig(n_features=8, d_token=16, n_heads=4, n_layers=1, use_gate=True)
    )
    spectral_model = TabularStateTransformer(
        TabularStateConfig(
            n_features=8,
            d_token=16,
            n_heads=4,
            n_layers=1,
            use_gate=True,
            use_fourier=True,
        )
    )
    assert gate_model(x).shape == (4,)
    assert spectral_model(x).shape == (4,)
