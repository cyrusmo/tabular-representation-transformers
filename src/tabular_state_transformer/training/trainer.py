from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tabular_state_transformer.blocks.gate import SparseFeatureGate
from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data.preprocessing import make_preprocessor, transform_to_float32
from tabular_state_transformer.data.schema import TabularDatasetBundle
from tabular_state_transformer.modeling import TabularStateTransformer
from tabular_state_transformer.training.diagnostics import (
    classify_effective_training_status,
    directional_final_vs_best,
    gate_summary,
    global_grad_norm,
    metric_improved,
    metric_mode,
    prediction_summary,
    train_val_gap,
)
from tabular_state_transformer.training.early_stopping import EarlyStopping
from tabular_state_transformer.training.losses import make_loss
from tabular_state_transformer.training.metrics import compute_metric
from tabular_state_transformer.utils.seed import seed_everything


@dataclass
class TrainingResult:
    model: nn.Module
    preprocessor: object
    train_loss: float
    val_metric: float
    processed_feature_names: list[str]
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_metric: float = 0.0
    final_val_metric: float = 0.0
    final_vs_best: float = 0.0
    effective_training_status: str = ""
    early_stopped: bool = False
    class_labels: np.ndarray | None = None


class Trainer:
    def __init__(
        self,
        config: TabularStateConfig,
        *,
        lr: float | None = None,
        batch_size: int | None = None,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float | None = None,
        model_factory: Callable[[TabularStateConfig], nn.Module] | None = None,
        device: str = "cpu",
    ):
        self.config = config
        self.lr = lr or config.learning_rate
        self.batch_size = batch_size or config.batch_size
        self.max_epochs = max_epochs or config.max_epochs
        self.early_stopping_patience = (
            config.early_stopping_patience if early_stopping_patience is None else early_stopping_patience
        )
        self.early_stopping_min_delta = (
            config.early_stopping_min_delta if early_stopping_min_delta is None else early_stopping_min_delta
        )
        self.model_factory = model_factory or TabularStateTransformer
        self.device = torch.device(device)

    def fit(self, bundle: TabularDatasetBundle) -> TrainingResult:
        seed_everything(self.config.random_state)
        preprocessor = make_preprocessor(bundle.X_train)
        X_train = transform_to_float32(preprocessor.fit(bundle.X_train), bundle.X_train)
        X_val = transform_to_float32(preprocessor, bundle.X_val)

        self.config.n_features = X_train.shape[1]
        self.config.task = bundle.task_type
        if bundle.task_type == "classification":
            classes, y_train = np.unique(bundle.y_train, return_inverse=True)
            y_val = np.searchsorted(classes, bundle.y_val)
            self.config.n_classes = len(classes)
            y_train_t = torch.as_tensor(y_train, dtype=torch.long)
            y_train_metric = np.asarray(y_train)
            y_val_metric = np.asarray(y_val)
        else:
            classes = None
            y_val = bundle.y_val
            y_train_t = torch.as_tensor(bundle.y_train, dtype=torch.float32)
            y_train_metric = np.asarray(bundle.y_train)
            y_val_metric = np.asarray(y_val)

        model = self.model_factory(self.config).to(self.device)
        loss_fn = make_loss(bundle.task_type)
        optim = torch.optim.AdamW(self._optimizer_param_groups(model), lr=self.lr)
        dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        last_loss = 0.0
        diagnostics: list[dict[str, object]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_val_metric: float | None = None
        early_stopped = False
        stopper = None
        if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
            stopper = EarlyStopping(
                patience=self.early_stopping_patience,
                mode=metric_mode(bundle.task_type),
                min_delta=self.early_stopping_min_delta,
            )
        train_eval_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        val_eval_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            batches = 0
            grad_norms: list[float] = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optim.zero_grad()
                output = model(xb)
                loss = loss_fn(output, yb)
                if self.config.use_gate and self.config.gate_l1:
                    loss = loss + self.config.gate_l1 * model.gate_regularization_loss()
                loss.backward()
                grad_norms.append(global_grad_norm(model.parameters()))
                optim.step()
                last_loss = float(loss.detach().cpu())
                epoch_loss += last_loss
                batches += 1

            train_loss = epoch_loss / max(batches, 1)
            model.eval()
            with torch.no_grad():
                train_output = model(train_eval_tensor)
                val_output = model(val_eval_tensor)
                train_metric = compute_metric(bundle.task_type, y_train_metric, train_output)
                val_metric = compute_metric(bundle.task_type, y_val_metric, val_output)

            if metric_improved(
                bundle.task_type,
                val_metric,
                best_val_metric,
                self.early_stopping_min_delta,
            ):
                best_val_metric = val_metric
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            row: dict[str, object] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
                "train_val_gap": train_val_gap(bundle.task_type, train_metric, val_metric),
            }
            row.update(prediction_summary(bundle.task_type, val_output))
            row.update(gate_summary(model))
            diagnostics.append(row)

            if stopper is not None and stopper.step(val_metric):
                early_stopped = True
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        final_val_metric = float(diagnostics[-1]["val_metric"]) if diagnostics else 0.0
        resolved_best_val_metric = float(best_val_metric if best_val_metric is not None else final_val_metric)
        final_vs_best = directional_final_vs_best(bundle.task_type, final_val_metric, resolved_best_val_metric)
        effective_training_status = classify_effective_training_status(
            diagnostics,
            task=bundle.task_type,
            best_epoch=best_epoch,
            best_val_metric=resolved_best_val_metric,
            early_stopped=early_stopped,
        )
        for row in diagnostics:
            row.update(
                {
                    "best_epoch": best_epoch,
                    "best_val_metric": resolved_best_val_metric,
                    "final_val_metric": final_val_metric,
                    "final_vs_best": final_vs_best,
                    "early_stopped": early_stopped,
                    "effective_training_status": effective_training_status,
                }
            )
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]
        return TrainingResult(
            model,
            preprocessor,
            last_loss,
            resolved_best_val_metric,
            feature_names,
            diagnostics,
            best_epoch,
            resolved_best_val_metric,
            final_val_metric,
            final_vs_best,
            effective_training_status,
            early_stopped,
            classes,
        )

    def _optimizer_param_groups(self, model: nn.Module) -> list[dict[str, object]]:
        if (
            not self.config.use_gate
            or self.config.gate_lr_multiplier == 1.0
            or not hasattr(model, "gate")
            or not isinstance(model.gate, SparseFeatureGate)
        ):
            return [{"params": list(model.parameters())}]

        gate_params = [model.gate.logits]
        gate_param_ids = {id(param) for param in gate_params}
        base_params = [param for param in model.parameters() if id(param) not in gate_param_ids]
        return [
            {"params": base_params},
            {"params": gate_params, "lr": self.lr * self.config.gate_lr_multiplier},
        ]
