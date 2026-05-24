from __future__ import annotations

import torch
from torch import nn

class SparseFeatureGate(nn.Module):
    """Learned per-feature gate. L1 regularization can be applied to regularization_loss."""

    def __init__(self, n_features: int, init: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_features,), float(init)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * torch.sigmoid(self.logits).view(1, -1, 1)

    def regularization_loss(self) -> torch.Tensor:
        gates = torch.sigmoid(self.logits)
        return torch.relu(gates - 0.1).sum()
