from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.evaluation.reporting import write_markdown_table
from tabular_state_transformer.models.baselines import make_baseline
from tabular_state_transformer.training import Trainer
from tabular_state_transformer.utils.io import read_yaml


@dataclass
class BenchmarkResult:
    model: str
    dataset: str
    metric: str
    score: float
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "metric": self.metric,
            "score": round(self.score, 6),
            "notes": self.notes,
        }


MODEL_CONFIGS = {
    "TST-v0": "configs/model/tst_v0.yaml",
    "TST-v1-Gate": "configs/model/tst_v1_gate.yaml",
    "TST-v2-GateFourier": "configs/model/tst_v2_fourier_gate.yaml",
}


def _metric(task: str, y_true: np.ndarray, pred) -> tuple[str, float]:
    if task == "classification":
        labels = np.asarray(pred).argmax(axis=1) if np.asarray(pred).ndim == 2 else pred
        return "accuracy", float(accuracy_score(y_true, labels))
    return "rmse", float(mean_squared_error(y_true, np.asarray(pred).reshape(-1)) ** 0.5)


def run_benchmark(
    suite: str = "synthetic",
    *,
    output_path: str | Path = "reports/benchmark_results.md",
    n_samples: int = 512,
    max_epochs: int = 2,
) -> list[BenchmarkResult]:
    dataset_names = ["synthetic_xor", "synthetic_piecewise"] if suite == "synthetic" else ["adult"]
    results: list[BenchmarkResult] = []
    repo_root = Path(__file__).resolve().parents[3]

    for dataset_name in dataset_names:
        load_kwargs = {"n_samples": n_samples} if dataset_name.startswith("synthetic") else {}
        bundle = load_dataset(dataset_name, split_seed=42, **load_kwargs)

        for baseline_name, label in [
            ("linear", "Linear/Ridge"),
            ("random_forest", "Random Forest"),
            ("mlp", "MLP"),
        ]:
            estimator = make_baseline(baseline_name, bundle.task_type, bundle.X_train)
            estimator.fit(bundle.X_train, bundle.y_train)
            pred = estimator.predict(bundle.X_test)
            metric, score = _metric(bundle.task_type, bundle.y_test, pred)
            results.append(BenchmarkResult(label, dataset_name, metric, score, "baseline"))

        for label, config_path in MODEL_CONFIGS.items():
            values = read_yaml(repo_root / config_path)
            values.update({"task": bundle.task_type, "max_epochs": max_epochs, "n_features": 1})
            config = TabularStateConfig.from_dict(values)
            trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
            X_test = trained.preprocessor.transform(bundle.X_test).astype("float32")
            pred = trained.model.predict_numpy(X_test)
            metric, score = _metric(bundle.task_type, bundle.y_test, pred)
            results.append(BenchmarkResult(label, dataset_name, metric, score, "ablation"))

    write_markdown_table([result.as_row() for result in results], output_path)
    return results
