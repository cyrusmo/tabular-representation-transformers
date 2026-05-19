from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.evaluation.reporting import write_csv_table, write_markdown_table
from tabular_state_transformer.models.baselines import make_baseline
from tabular_state_transformer.training import Trainer
from tabular_state_transformer.utils.io import read_yaml


@dataclass
class BenchmarkResult:
    model: str
    dataset: str
    seed: int
    task: str
    family: str
    variant: str
    metric: str
    score: float
    fit_seconds: float
    predict_seconds: float
    n_samples: int
    n_features: int
    status: str = "ok"
    error_message: str = ""
    artifact_path: str = ""
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "seed": self.seed,
            "task": self.task,
            "family": self.family,
            "variant": self.variant,
            "status": self.status,
            "metric": self.metric,
            "score": round(self.score, 6),
            "fit_seconds": round(self.fit_seconds, 4),
            "predict_seconds": round(self.predict_seconds, 4),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "artifact_path": self.artifact_path,
            "error_message": self.error_message,
            "notes": self.notes,
        }


SYNTHETIC_STRESS_DATASETS = [
    "synthetic_axis_aligned",
    "synthetic_xor",
    "synthetic_piecewise",
    "synthetic_irrelevant_noise",
    "synthetic_rotated",
    "synthetic_regime",
    "synthetic_sparse_high_order",
]

BASELINE_CONFIGS = {
    "linear": "Linear/Ridge",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

DEFAULT_BASELINES = [
    "linear",
    "random_forest",
    "gradient_boosting",
    "mlp",
    "lightgbm",
    "catboost",
]

MODEL_CONFIGS = {
    "TST-v0": "configs/model/tst_v0.yaml",
    "TST-v1-Gate": "configs/model/tst_v1_gate.yaml",
    "TST-v2-GateFourier": "configs/model/tst_v2_fourier_gate.yaml",
    "TST-v3-MoE": "configs/model/tst_v3_moe.yaml",
}


def _metric(task: str, y_true: np.ndarray, pred) -> tuple[str, float]:
    if task == "classification":
        labels = np.asarray(pred).argmax(axis=1) if np.asarray(pred).ndim == 2 else pred
        return "accuracy", float(accuracy_score(y_true, labels))
    return "rmse", float(mean_squared_error(y_true, np.asarray(pred).reshape(-1)) ** 0.5)


def _datasets_for_suite(suite: str) -> list[str]:
    if suite == "synthetic":
        return ["synthetic_xor", "synthetic_piecewise"]
    if suite == "synthetic_stress":
        return SYNTHETIC_STRESS_DATASETS
    if suite == "openml":
        return ["adult"]
    raise ValueError(f"Unknown benchmark suite '{suite}'.")


def _error_result(
    *,
    model: str,
    dataset: str,
    seed: int,
    task: str,
    family: str,
    variant: str,
    n_samples: int,
    n_features: int,
    error: Exception,
) -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        dataset=dataset,
        seed=seed,
        task=task,
        family=family,
        variant=variant,
        metric="error",
        score=float("nan"),
        fit_seconds=0.0,
        predict_seconds=0.0,
        n_samples=n_samples,
        n_features=n_features,
        status="error",
        error_message=str(error),
    )


def run_benchmark(
    suite: str = "synthetic",
    *,
    output_path: str | Path = "reports/benchmark_results.md",
    csv_output_path: str | Path | None = None,
    n_samples: int = 512,
    max_epochs: int = 2,
    seeds: Sequence[int] | None = None,
    dataset_names: Sequence[str] | None = None,
    baselines: Sequence[str] | None = None,
    model_configs: Sequence[str] | None = None,
    continue_on_error: bool = True,
) -> list[BenchmarkResult]:
    selected_datasets = list(_datasets_for_suite(suite) if dataset_names is None else dataset_names)
    selected_seeds = list([42] if seeds is None else seeds)
    selected_baselines = list(DEFAULT_BASELINES if baselines is None else baselines)
    selected_models = list(MODEL_CONFIGS if model_configs is None else model_configs)
    results: list[BenchmarkResult] = []
    repo_root = Path(__file__).resolve().parents[3]

    for seed in selected_seeds:
        for dataset_name in selected_datasets:
            load_kwargs = {"n_samples": n_samples} if dataset_name.startswith("synthetic") else {}
            bundle = load_dataset(dataset_name, split_seed=seed, **load_kwargs)
            total_samples = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
            raw_features = bundle.X_train.shape[1]

            for baseline_name in selected_baselines:
                label = BASELINE_CONFIGS[baseline_name]
                try:
                    start = perf_counter()
                    estimator = make_baseline(
                        baseline_name,
                        bundle.task_type,
                        bundle.X_train,
                        random_state=seed,
                    )
                    estimator.fit(bundle.X_train, bundle.y_train)
                    fit_seconds = perf_counter() - start
                    start = perf_counter()
                    pred = estimator.predict(bundle.X_test)
                    predict_seconds = perf_counter() - start
                    metric, score = _metric(bundle.task_type, bundle.y_test, pred)
                    results.append(
                        BenchmarkResult(
                            label,
                            dataset_name,
                            seed,
                            bundle.task_type,
                            "baseline",
                            baseline_name,
                            metric,
                            score,
                            fit_seconds,
                            predict_seconds,
                            total_samples,
                            raw_features,
                        )
                    )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    results.append(
                        _error_result(
                            model=label,
                            dataset=dataset_name,
                            seed=seed,
                            task=bundle.task_type,
                            family="baseline",
                            variant=baseline_name,
                            n_samples=total_samples,
                            n_features=raw_features,
                            error=exc,
                        )
                    )

            for label in selected_models:
                try:
                    values = read_yaml(repo_root / MODEL_CONFIGS[label])
                    values.update(
                        {
                            "task": bundle.task_type,
                            "max_epochs": max_epochs,
                            "n_features": 1,
                            "random_state": seed,
                        }
                    )
                    config = TabularStateConfig.from_dict(values)
                    start = perf_counter()
                    trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
                    fit_seconds = perf_counter() - start
                    X_test = trained.preprocessor.transform(bundle.X_test).astype("float32")
                    start = perf_counter()
                    pred = trained.model.predict_numpy(X_test)
                    predict_seconds = perf_counter() - start
                    metric, score = _metric(bundle.task_type, bundle.y_test, pred)
                    results.append(
                        BenchmarkResult(
                            label,
                            dataset_name,
                            seed,
                            bundle.task_type,
                            "ablation",
                            label,
                            metric,
                            score,
                            fit_seconds,
                            predict_seconds,
                            total_samples,
                            raw_features,
                        )
                    )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    results.append(
                        _error_result(
                            model=label,
                            dataset=dataset_name,
                            seed=seed,
                            task=bundle.task_type,
                            family="ablation",
                            variant=label,
                            n_samples=total_samples,
                            n_features=raw_features,
                            error=exc,
                        )
                    )

    write_markdown_table([result.as_row() for result in results], output_path)
    if csv_output_path is not None:
        write_csv_table([result.as_row() for result in results], csv_output_path)
    return results
