from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from .config import TabularStateConfig
from .modeling import TabularStateTransformer

class _BaseTabularState(BaseEstimator):
    def __init__(
        self,
        d_token: int = 64,
        max_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.d_token = d_token
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def _fit_torch(self, X, y, task: str, n_classes: int = 2):
        torch.manual_seed(self.random_state)
        self.scaler_ = StandardScaler().fit(X)
        X_t = torch.tensor(self.scaler_.transform(X), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long if task == "classification" else torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
        self.model_ = TabularStateTransformer(
            TabularStateConfig(
                n_features=X_t.shape[1],
                d_token=self.d_token,
                task=task,
                n_classes=n_classes,
            )
        )
        optim = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.model_.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                optim.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optim.step()
        return self

class TabularStateRegressor(_BaseTabularState, RegressorMixin):
    def fit(self, X, y):
        return self._fit_torch(np.asarray(X), np.asarray(y), task="regression")
    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler_.transform(np.asarray(X)), dtype=torch.float32)
            return self.model_(x).cpu().numpy()

class TabularStateClassifier(_BaseTabularState, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y)
        return self._fit_torch(np.asarray(X), y_encoded, task="classification", n_classes=len(self.classes_))
    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler_.transform(np.asarray(X)), dtype=torch.float32)
            return torch.softmax(self.model_(x), dim=-1).cpu().numpy()
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
