from __future__ import annotations

import torch
from torch import nn

class FourierFeatureBlock(nn.Module):
    """Controlled high-frequency expansion over token channels."""
    def __init__(self, d_token: int, n_frequencies: int = 4):
        super().__init__()
        self.register_buffer("freq", 2 ** torch.arange(n_frequencies).float())
        self.proj = nn.Linear(d_token * (1 + 2 * n_frequencies), d_token)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expanded = [tokens]
        for f in self.freq:
            expanded.append(torch.sin(f * tokens))
            expanded.append(torch.cos(f * tokens))
        return self.proj(torch.cat(expanded, dim=-1))

class WaveletFeatureBlock(nn.Module):
    """Lightweight Haar-style local contrast block over feature-token order."""
    def __init__(self, d_token: int):
        super().__init__()
        self.proj = nn.Linear(d_token * 2, d_token)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        shifted = torch.roll(tokens, shifts=1, dims=1)
        contrast = tokens - shifted
        return self.proj(torch.cat([tokens, contrast], dim=-1))
