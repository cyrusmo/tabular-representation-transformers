from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Iterable

import numpy as np
import torch

from tabular_state_transformer.blocks.gate import SparseFeatureGate
from tabular_state_transformer.modeling import TabularStateTransformer

STATUS_MIN_LOSS_IMPROVEMENT = 1e-4
STATUS_MIN_METRIC_IMPROVEMENT = 1e-4
STATUS_TRAIN_VAL_GAP_THRESHOLD = 0.15
STATUS_EXTREME_GRAD_NORM = 1e4
STATUS_EXTREME_PREDICTION_MAGNITUDE = 1e6
STATUS_BEST_NEAR_FINAL_FRACTION = 0.2
STATUS_POOR_CLASSIFICATION_METRIC = 0.7
STATUS_POOR_REGRESSION_RMSE = 0.75
GATE_SPARSITY_THRESHOLD = 0.1


@dataclass(frozen=True)
class DiagnosticThresholds:
    min_loss_improvement: float = STATUS_MIN_LOSS_IMPROVEMENT
    min_metric_improvement: float = STATUS_MIN_METRIC_IMPROVEMENT
    train_val_gap: float = STATUS_TRAIN_VAL_GAP_THRESHOLD
    extreme_grad_norm: float = STATUS_EXTREME_GRAD_NORM
    extreme_prediction_magnitude: float = STATUS_EXTREME_PREDICTION_MAGNITUDE
    best_near_final_fraction: float = STATUS_BEST_NEAR_FINAL_FRACTION
    poor_classification_metric: float = STATUS_POOR_CLASSIFICATION_METRIC
    poor_regression_rmse: float = STATUS_POOR_REGRESSION_RMSE


DEFAULT_THRESHOLDS = DiagnosticThresholds()


def metric_mode(task: str) -> str:
    return "max" if task == "classification" else "min"


def metric_improved(task: str, current: float, best: float | None, min_delta: float) -> bool:
    if best is None:
        return True
    if task == "classification":
        return current > best + min_delta
    return current < best - min_delta


def metric_gain(task: str, initial: float, final: float) -> float:
    if task == "classification":
        return final - initial
    return initial - final


def directional_final_vs_best(task: str, final_metric: float, best_metric: float) -> float:
    if task == "classification":
        return best_metric - final_metric
    return final_metric - best_metric


def train_val_gap(task: str, train_metric: float, val_metric: float) -> float:
    if task == "classification":
        return train_metric - val_metric
    return val_metric - train_metric


def prediction_summary(task: str, output: torch.Tensor) -> dict[str, float]:
    values = output.detach().cpu()
    if task == "classification":
        probs = torch.softmax(values, dim=-1)
        if probs.ndim == 2 and probs.shape[1] > 1:
            values = probs[:, 1]
        else:
            values = probs.reshape(-1)
    else:
        values = values.reshape(-1)
    return {
        "prediction_mean": float(values.mean().item()),
        "prediction_std": float(values.std(unbiased=False).item()),
    }


def extract_gate_values(
    model: TabularStateTransformer,
    X_valid: np.ndarray | None = None,
) -> np.ndarray | None:
    if isinstance(model.gate, SparseFeatureGate):
        return torch.sigmoid(model.gate.logits.detach()).cpu().numpy()
    return None


def gate_summary(model: TabularStateTransformer) -> dict[str, object]:
    values = extract_gate_values(model)
    if values is None:
        return {
            "has_gate": False,
            "gate_mean": "",
            "gate_median": "",
            "gate_sparsity": "",
        }
    return {
        "has_gate": True,
        "gate_mean": float(np.mean(values)),
        "gate_median": float(np.median(values)),
        "gate_sparsity": float(np.mean(values < GATE_SPARSITY_THRESHOLD)),
    }


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().data.norm(2).cpu().item())
        total += value * value
    return total**0.5


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and isfinite(float(value))


def _has_unstable_values(
    history: list[dict[str, object]],
    thresholds: DiagnosticThresholds,
) -> bool:
    numeric_keys = [
        "train_loss",
        "train_metric",
        "val_metric",
        "grad_norm",
        "prediction_mean",
        "prediction_std",
    ]
    for row in history:
        for key in numeric_keys:
            value = row.get(key)
            if not _finite_number(value):
                return True
        if float(row["grad_norm"]) > thresholds.extreme_grad_norm:
            return True
        if abs(float(row["prediction_mean"])) > thresholds.extreme_prediction_magnitude:
            return True
        if abs(float(row["prediction_std"])) > thresholds.extreme_prediction_magnitude:
            return True
    return False


def _best_epoch_near_final(history: list[dict[str, object]], best_epoch: int, thresholds: DiagnosticThresholds) -> bool:
    final_epoch = int(history[-1]["epoch"])
    window = max(1, ceil(final_epoch * thresholds.best_near_final_fraction))
    return best_epoch >= final_epoch - window + 1


def classify_effective_training_status(
    history: list[dict[str, object]],
    *,
    task: str,
    best_epoch: int,
    best_val_metric: float,
    early_stopped: bool,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> str:
    if not history or _has_unstable_values(history, thresholds):
        return "unstable"

    first = history[0]
    final = history[-1]
    loss_improvement = float(first["train_loss"]) - min(float(row["train_loss"]) for row in history)
    train_improvement = metric_gain(task, float(first["train_metric"]), float(final["train_metric"]))
    val_improvement = metric_gain(task, float(first["val_metric"]), float(final["val_metric"]))
    final_gap = train_val_gap(task, float(final["train_metric"]), float(final["val_metric"]))

    if (
        loss_improvement <= thresholds.min_loss_improvement
        and train_improvement <= thresholds.min_metric_improvement
    ):
        return "no_learning"

    if (train_improvement > thresholds.min_metric_improvement and val_improvement < -thresholds.min_metric_improvement) or (
        final_gap > thresholds.train_val_gap
    ):
        return "overfit"

    if task == "classification":
        train_poor = float(final["train_metric"]) < thresholds.poor_classification_metric
        val_poor = best_val_metric < thresholds.poor_classification_metric
    else:
        train_poor = float(final["train_metric"]) > thresholds.poor_regression_rmse
        val_poor = best_val_metric > thresholds.poor_regression_rmse

    if train_poor and val_poor and _best_epoch_near_final(history, best_epoch, thresholds):
        return "underfit"

    if early_stopped:
        return "early_stopped_cleanly"

    return "underfit"
