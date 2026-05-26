from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .blocks.feature_cross import FeatureCrossTokenizer
from .blocks.gate import SparseFeatureGate
from .blocks.heads import ClassificationHead, RegressionHead
from .blocks.interaction import InteractionBlock
from .blocks.moe_head import RegimeGatedHead
from .blocks.pooling import AttentionPooling
from .blocks.spectral import FourierFeatureBlock, WaveletFeatureBlock
from .config import TabularStateConfig
from .tokenizer import FeatureTokenizer

class TabularStateTransformer(nn.Module):
    def __init__(self, config: TabularStateConfig):
        super().__init__()
        if config.pooling not in {"mean", "cls", "attention"}:
            raise ValueError(f"Unknown pooling mode '{config.pooling}'.")
        self.config = config
        self.tokenizer = FeatureTokenizer(config.n_features, config.d_token)
        self.gate = SparseFeatureGate(config.n_features, init=config.gate_init) if config.use_gate else nn.Identity()
        self.fourier = FourierFeatureBlock(config.d_token) if config.use_fourier else nn.Identity()
        self.wavelet = WaveletFeatureBlock(config.d_token) if config.use_wavelet else nn.Identity()
        self.cross_tokenizer = (
            FeatureCrossTokenizer(config.n_features, config.d_token, config.cross_max_features)
            if config.use_feature_crosses
            else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_token)) if config.pooling == "cls" else None
        self.attention_pooling = AttentionPooling(config.d_token) if config.pooling == "attention" else None
        self.interaction = InteractionBlock(config.d_token, config.n_heads, config.n_layers, config.dropout)
        output_dim = config.n_classes if config.task == "classification" else 1
        if config.use_moe or config.head_type == "moe":
            self.head = RegimeGatedHead(config.d_token, output_dim, config.n_experts)
        elif config.task == "classification":
            self.head = ClassificationHead(config.d_token, config.n_classes)
        else:
            self.head = RegressionHead(config.d_token)

    def _tokens_before_interaction(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        tokens = self.tokenizer(x.float())
        tokens = self.gate(tokens)
        tokens = self.fourier(tokens)
        tokens = self.wavelet(tokens)
        if self.cross_tokenizer is not None:
            tokens = torch.cat([tokens, self.cross_tokenizer(x)], dim=1)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        return tokens

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "mean":
            return tokens.mean(dim=1)
        if self.config.pooling == "cls":
            return tokens[:, 0]
        if self.attention_pooling is None:
            raise ValueError(f"Unknown pooling mode '{self.config.pooling}'.")
        return self.attention_pooling(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens_before_interaction(x)
        tokens = self.interaction(tokens)
        out = self.head(self._pool_tokens(tokens))
        return out if self.config.task == "classification" else out.squeeze(-1)

    def gate_regularization_loss(self) -> torch.Tensor:
        if isinstance(self.gate, SparseFeatureGate):
            return self.gate.regularization_loss()
        return torch.zeros((), device=next(self.parameters()).device)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            output = self(tensor)
            if self.config.task == "classification":
                output = torch.softmax(output, dim=-1)
            return output.cpu().numpy()

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        state_dict = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        try:
            from safetensors.torch import save_file

            save_file(state_dict, save_path / "model.safetensors")
        except ImportError:
            torch.save(state_dict, save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> "TabularStateTransformer":
        load_path = Path(load_directory)
        config = TabularStateConfig.from_dict(json.loads((load_path / "config.json").read_text()))
        model = cls(config)
        safetensors_path = load_path / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(load_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model
