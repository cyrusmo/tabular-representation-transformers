from __future__ import annotations

import torch
from torch import nn

class InteractionBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
