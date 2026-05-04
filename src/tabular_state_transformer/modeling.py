from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .blocks.gate import SparseFeatureGate
from .blocks.heads import ClassificationHead, RegressionHead
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
        self.gate = SparseFeatureGate(config.n_features) if config.use_gate else nn.Identity()
        self.fourier = FourierFeatureBlock(config.d_token) if config.use_fourier else nn.Identity()
        self.wavelet = WaveletFeatureBlock(config.d_token) if config.use_wavelet else nn.Identity()
        self.interaction = InteractionBlock(config.d_token, config.n_heads, config.n_layers, config.dropout)
        output_dim = config.n_classes if config.task == "classification" else 1
        if config.use_moe or config.head_type == "moe":
            self.head = RegimeGatedHead(config.d_token, output_dim, config.n_experts)
        elif config.task == "classification":
            self.head = ClassificationHead(config.d_token, config.n_classes)
        else:
            self.head = RegressionHead(config.d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x.float())
        tokens = self.gate(tokens)
        tokens = self.fourier(tokens)
        tokens = self.wavelet(tokens)
        tokens = self.interaction(tokens)
        out = self.head(tokens.mean(dim=1))
        return out if self.config.task == "classification" else out.squeeze(-1)

    def gate_regularization_loss(self) -> torch.Tensor:
        if isinstance(self.gate, SparseFeatureGate):
            return self.gate.regularization_loss()
        return torch.zeros((), device=next(self.parameters()).device)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32)
            output = self(tensor)
            if self.config.task == "classification":
                output = torch.softmax(output, dim=-1)
            return output.cpu().numpy()

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        try:
            from safetensors.torch import save_file

            save_file(self.state_dict(), save_path / "model.safetensors")
        except ImportError:
            torch.save(self.state_dict(), save_path / "pytorch_model.bin")

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
