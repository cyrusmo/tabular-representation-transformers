from __future__ import annotations

import numpy as np
import pytest
import torch

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.data.preprocessing import transform_to_float32
from tabular_state_transformer.training import Trainer
from tabular_state_transformer.training.diagnostics import classify_effective_training_status
from tabular_state_transformer.training.early_stopping import EarlyStopping
from tabular_state_transformer.training.metrics import compute_metric


def _row(
    *,
    epoch: int,
    train_loss: float,
    train_metric: float,
    val_metric: float,
    grad_norm: float = 1.0,
    prediction_mean: float = 0.5,
    prediction_std: float = 0.1,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_metric": train_metric,
        "val_metric": val_metric,
        "grad_norm": grad_norm,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
    }


def test_effective_training_status_examples_are_deterministic():
    assert (
        classify_effective_training_status(
            [_row(epoch=1, train_loss=1.0, train_metric=0.5, val_metric=0.5, grad_norm=1e5)],
            task="classification",
            best_epoch=1,
            best_val_metric=0.5,
            early_stopped=False,
        )
        == "unstable"
    )
    assert (
        classify_effective_training_status(
            [
                _row(epoch=1, train_loss=1.0, train_metric=0.5, val_metric=0.5),
                _row(epoch=2, train_loss=1.0, train_metric=0.5, val_metric=0.5),
            ],
            task="classification",
            best_epoch=2,
            best_val_metric=0.5,
            early_stopped=False,
        )
        == "no_learning"
    )
    assert (
        classify_effective_training_status(
            [
                _row(epoch=1, train_loss=1.0, train_metric=0.5, val_metric=0.8),
                _row(epoch=2, train_loss=0.2, train_metric=0.9, val_metric=0.6),
            ],
            task="classification",
            best_epoch=1,
            best_val_metric=0.8,
            early_stopped=False,
        )
        == "overfit"
    )
    assert (
        classify_effective_training_status(
            [
                _row(epoch=1, train_loss=1.0, train_metric=0.5, val_metric=0.5),
                _row(epoch=2, train_loss=0.8, train_metric=0.6, val_metric=0.6),
            ],
            task="classification",
            best_epoch=2,
            best_val_metric=0.6,
            early_stopped=False,
        )
        == "underfit"
    )
    assert (
        classify_effective_training_status(
            [
                _row(epoch=1, train_loss=1.0, train_metric=0.5, val_metric=0.6),
                _row(epoch=2, train_loss=0.6, train_metric=0.75, val_metric=0.8),
                _row(epoch=3, train_loss=0.5, train_metric=0.8, val_metric=0.75),
            ],
            task="classification",
            best_epoch=2,
            best_val_metric=0.8,
            early_stopped=True,
        )
        == "early_stopped_cleanly"
    )


def test_early_stopping_respects_min_delta():
    stopper = EarlyStopping(patience=2, mode="max", min_delta=0.01)
    assert stopper.step(0.5) is False
    assert stopper.step(0.505) is False
    assert stopper.step(0.506) is True


def test_trainer_emits_diagnostics_and_restores_best_checkpoint():
    bundle = load_dataset("synthetic_xor", split_seed=42, n_samples=128, n_features=8)
    config = TabularStateConfig(
        n_features=8,
        d_token=8,
        n_heads=2,
        n_layers=1,
        task="classification",
        max_epochs=3,
        batch_size=32,
        early_stopping_patience=None,
    )
    result = Trainer(config).fit(bundle)

    assert result.diagnostics
    assert result.best_epoch >= 1
    assert result.effective_training_status
    assert all(row["effective_training_status"] for row in result.diagnostics)
    assert all(np.isfinite(float(row["train_loss"])) for row in result.diagnostics)

    X_val = transform_to_float32(result.preprocessor, bundle.X_val)
    with torch.no_grad():
        output = result.model(torch.as_tensor(X_val, dtype=torch.float32))
    restored_metric = compute_metric(bundle.task_type, bundle.y_val, output)
    assert restored_metric == pytest.approx(result.best_val_metric)
