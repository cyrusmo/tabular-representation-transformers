from __future__ import annotations

import torch
from torch import nn

class FeatureTokenizer(nn.Module):
    """Numeric feature tokenizer that preserves column identity via learned embeddings."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.column_embedding = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) + self.column_embedding.unsqueeze(0)
