from __future__ import annotations

import torch
from torch import nn

class RegimeGatedHead(nn.Module):
    def __init__(self, d_token: int, output_dim: int, n_experts: int = 3):
        super().__init__()
        self.gate = nn.Linear(d_token, n_experts)
        self.experts = nn.ModuleList([nn.Linear(d_token, output_dim) for _ in range(n_experts)])
    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(pooled), dim=-1)
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
