from __future__ import annotations

from dataclasses import asdict, dataclass

@dataclass
class TabularStateConfig:
    n_features: int
    d_token: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_experts: int = 3
    use_sparse_gate: bool = True
    use_fourier: bool = True
    use_wavelet: bool = False
    task: str = "regression"
    n_classes: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)
