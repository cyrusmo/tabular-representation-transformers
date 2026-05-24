from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class TabularStateConfig:
    n_features: int
    d_token: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_experts: int = 3
    use_gate: bool = False
    use_sparse_gate: bool | None = None
    use_fourier: bool = False
    use_wavelet: bool = False
    use_moe: bool = False
    head_type: str = "simple"
    task: str = "regression"
    n_classes: int = 2
    dropout: float = 0.1
    gate_l1: float = 0.0
    gate_init: float = -1.0
    gate_lr_multiplier: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 20
    random_state: int = 42
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self) -> None:
        if self.use_sparse_gate is not None:
            self.use_gate = self.use_sparse_gate
        self.use_sparse_gate = self.use_gate
        if self.use_moe:
            self.head_type = "moe"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TabularStateConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in values.items() if k in allowed})
