from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.steps = 0

    def step(self, value: float) -> bool:
        improved = self.best is None or (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.steps = 0
            return False
        self.steps += 1
        return self.steps >= self.patience
