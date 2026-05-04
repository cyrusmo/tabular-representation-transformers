from __future__ import annotations

import torch

from tabular_state_transformer.tokenizer import FeatureTokenizer


def test_tokenizer_output_shape():
    tokenizer = FeatureTokenizer(n_features=5, d_token=16)
    output = tokenizer(torch.randn(7, 5))
    assert output.shape == (7, 5, 16)
