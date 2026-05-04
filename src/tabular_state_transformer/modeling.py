from __future__ import annotations

import torch
from torch import nn
from .blocks.gate import SparseFeatureGate
from .blocks.interaction import InteractionBlock
from .blocks.moe_head import RegimeGatedHead
from .blocks.spectral import FourierFeatureBlock, WaveletFeatureBlock
from .config import TabularStateConfig
from .tokenizer import FeatureTokenizer

class TabularStateTransformer(nn.Module):
    def __init__(self, config: TabularStateConfig):
        super().__init__()
        self.config = config
        self.tokenizer = FeatureTokenizer(config.n_features, config.d_token)
        self.gate = SparseFeatureGate(config.n_features) if config.use_sparse_gate else nn.Identity()
        self.fourier = FourierFeatureBlock(config.d_token) if config.use_fourier else nn.Identity()
        self.wavelet = WaveletFeatureBlock(config.d_token) if config.use_wavelet else nn.Identity()
        self.interaction = InteractionBlock(config.d_token, config.n_heads, config.n_layers, config.dropout)
        output_dim = config.n_classes if config.task == "classification" else 1
        self.head = RegimeGatedHead(config.d_token, output_dim, config.n_experts)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x.float())
        tokens = self.gate(tokens)
        tokens = self.fourier(tokens)
        tokens = self.wavelet(tokens)
        tokens = self.interaction(tokens)
        out = self.head(tokens.mean(dim=1))
        return out if self.config.task == "classification" else out.squeeze(-1)
