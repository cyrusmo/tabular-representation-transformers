from __future__ import annotations

import math

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, d_token: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_token) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = tokens @ self.query / math.sqrt(tokens.shape[-1])
        weights = torch.softmax(scores, dim=1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)
