from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error


def compute_metric(task: str, y_true: np.ndarray, output: torch.Tensor) -> float:
    if task == "classification":
        pred = output.detach().cpu().numpy().argmax(axis=1)
        return float(accuracy_score(y_true, pred))
    pred = output.detach().cpu().numpy().reshape(-1)
    return float(mean_squared_error(y_true, pred) ** 0.5)
