from __future__ import annotations

from torch import nn

class RegressionHead(nn.Sequential):
    def __init__(self, d_token: int):
        super().__init__(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

class ClassificationHead(nn.Sequential):
    def __init__(self, d_token: int, n_classes: int):
        super().__init__(nn.LayerNorm(d_token), nn.Linear(d_token, n_classes))
