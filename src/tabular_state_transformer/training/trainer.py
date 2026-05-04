from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data.preprocessing import make_preprocessor, transform_to_float32
from tabular_state_transformer.data.schema import TabularDatasetBundle
from tabular_state_transformer.modeling import TabularStateTransformer
from tabular_state_transformer.training.losses import make_loss
from tabular_state_transformer.training.metrics import compute_metric
from tabular_state_transformer.utils.seed import seed_everything


@dataclass
class TrainingResult:
    model: TabularStateTransformer
    preprocessor: object
    train_loss: float
    val_metric: float
    processed_feature_names: list[str]


class Trainer:
    def __init__(
        self,
        config: TabularStateConfig,
        *,
        lr: float | None = None,
        batch_size: int | None = None,
        max_epochs: int | None = None,
        device: str = "cpu",
    ):
        self.config = config
        self.lr = lr or config.learning_rate
        self.batch_size = batch_size or config.batch_size
        self.max_epochs = max_epochs or config.max_epochs
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
        else:
            y_val = bundle.y_val
            y_train_t = torch.as_tensor(bundle.y_train, dtype=torch.float32)

        model = TabularStateTransformer(self.config).to(self.device)
        loss_fn = make_loss(bundle.task_type)
        optim = torch.optim.AdamW(model.parameters(), lr=self.lr)
        dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        last_loss = 0.0

        model.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optim.zero_grad()
                output = model(xb)
                loss = loss_fn(output, yb)
                if self.config.use_gate and self.config.gate_l1:
                    loss = loss + self.config.gate_l1 * model.gate_regularization_loss()
                loss.backward()
                optim.step()
                last_loss = float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            val_output = model(torch.as_tensor(X_val, dtype=torch.float32, device=self.device))
            val_metric = compute_metric(bundle.task_type, np.asarray(y_val), val_output)
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]
        return TrainingResult(model, preprocessor, last_loss, val_metric, feature_names)
