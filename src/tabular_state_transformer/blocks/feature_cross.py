from __future__ import annotations

import itertools

import torch
from torch import nn


class FeatureCrossTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int, max_features: int = 16):
        super().__init__()
        self.n_features = n_features
        self.max_features = min(n_features, max_features)
        pairs = list(itertools.combinations(range(self.max_features), 2))
        self.register_buffer("pair_indices", torch.as_tensor(pairs, dtype=torch.long))
        n_pairs = len(pairs)
        self.pair_weight_embedding = nn.Parameter(torch.randn(n_pairs, d_token) * 0.02)
        self.pair_bias_embedding = nn.Parameter(torch.zeros(n_pairs, d_token))
        self.pair_identity_embedding = nn.Parameter(torch.randn(n_pairs, d_token) * 0.02)

    @property
    def num_cross_tokens(self) -> int:
        return int(self.pair_indices.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_cross_tokens == 0:
            return x.new_zeros((x.shape[0], 0, self.pair_weight_embedding.shape[-1]))
        left = self.pair_indices[:, 0]
        right = self.pair_indices[:, 1]
        cross_values = x[:, left] * x[:, right]
        return (
            cross_values.unsqueeze(-1) * self.pair_weight_embedding.unsqueeze(0)
            + self.pair_bias_embedding.unsqueeze(0)
            + self.pair_identity_embedding.unsqueeze(0)
        )
