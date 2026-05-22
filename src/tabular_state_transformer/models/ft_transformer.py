from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tabular_state_transformer.blocks.heads import ClassificationHead, RegressionHead
from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.tokenizer import FeatureTokenizer


class FTTransformerStyle(nn.Module):
    """Local FT-Transformer-style baseline for numeric preprocessed tabular features."""

    def __init__(self, config: TabularStateConfig):
        super().__init__()
        self.config = config
        self.tokenizer = FeatureTokenizer(config.n_features, config.d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_token,
            nhead=config.n_heads,
            dim_feedforward=config.d_token * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.gate = nn.Identity()
        output_dim = config.n_classes if config.task == "classification" else 1
        self.head = (
            ClassificationHead(config.d_token, output_dim)
            if config.task == "classification"
            else RegressionHead(config.d_token)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x.float())
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        output = self.head(encoded[:, 0])
        return output if self.config.task == "classification" else output.squeeze(-1)

    def gate_regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            output = self(torch.as_tensor(x, dtype=torch.float32))
            if self.config.task == "classification":
                output = torch.softmax(output, dim=-1)
            return output.cpu().numpy()
