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

TREE_BASELINE_LABELS = {"Random Forest", "Gradient Boosting", "LightGBM", "CatBoost"}

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


def _ok_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    return [result for result in results if result.status == "ok" and np.isfinite(result.score)]


def _ranked_results(results: list[BenchmarkResult]) -> list[tuple[BenchmarkResult, int]]:
    groups: dict[tuple[str, int, str], list[BenchmarkResult]] = {}
    for result in _ok_results(results):
        groups.setdefault((result.dataset, result.seed, result.metric), []).append(result)

    ranked: list[tuple[BenchmarkResult, int]] = []
    for (_, _, metric), group in groups.items():
        reverse = metric == "accuracy"
        ordered = sorted(group, key=lambda result: result.score, reverse=reverse)
        previous_score: float | None = None
        previous_rank = 0
        for index, result in enumerate(ordered, start=1):
            if previous_score is not None and result.score == previous_score:
                rank = previous_rank
            else:
                rank = index
                previous_rank = rank
                previous_score = result.score
            ranked.append((result, rank))
    return ranked


def _append_benchmark_summary(results: list[BenchmarkResult], output_path: str | Path) -> None:
    path = Path(output_path)
    ranked = _ranked_results(results)
    ok_count = len(_ok_results(results))
    error_count = len(results) - ok_count
    by_model: dict[str, dict[str, object]] = {}
    for result, rank in ranked:
        stats = by_model.setdefault(
            result.model,
            {
                "family": result.family,
                "variant": result.variant,
                "ranks": [],
                "wins": 0,
                "ok_rows": 0,
            },
        )
        stats["ranks"].append(rank)
        stats["ok_rows"] = int(stats["ok_rows"]) + 1
        if rank == 1:
            stats["wins"] = int(stats["wins"]) + 1

    summary_rows = []
    for model, stats in by_model.items():
        ranks = stats["ranks"]
        assert isinstance(ranks, list)
        summary_rows.append(
            {
                "model": model,
                "family": stats["family"],
                "variant": stats["variant"],
                "mean_rank": sum(ranks) / len(ranks),
                "wins": stats["wins"],
                "ok_rows": stats["ok_rows"],
            }
        )
    summary_rows.sort(key=lambda row: (float(row["mean_rank"]), str(row["model"])))

    tree_ranks = [rank for result, rank in ranked if result.model in TREE_BASELINE_LABELS]
    tst_ranks = [rank for result, rank in ranked if result.family == "ablation"]
    tree_mean = sum(tree_ranks) / len(tree_ranks) if tree_ranks else float("nan")
    tst_mean = sum(tst_ranks) / len(tst_ranks) if tst_ranks else float("nan")
    if np.isfinite(tree_mean) and np.isfinite(tst_mean) and tree_mean < tst_mean:
        interpretation = (
            "Tree baselines remain ahead on this refresh. That is the empirical baseline for the "
            "next diagnostic pass, not a result to hide."
        )
    elif np.isfinite(tree_mean) and np.isfinite(tst_mean):
        interpretation = (
            "TST variants are competitive with the tree baselines on mean rank in this refresh; "
            "the next step is to confirm whether diagnostics show stable learning."
        )
    else:
        interpretation = "Tree-vs-TST comparison is unavailable because one side has no successful rows."

    lines = [
        "",
        "## Rank And Win Summary",
        "",
        f"- Total rows: {len(results)}",
        f"- Successful rows: {ok_count}",
        f"- Error rows: {error_count}",
        "- Lower mean rank is better; `status=error` rows are excluded from ranks and wins.",
        "",
        "| Model | Family | Variant | Mean Rank | Wins | Ok Rows |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {family} | {variant} | {mean_rank:.3f} | {wins} | {ok_rows} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Tree Vs TST Interpretation",
            "",
            f"- Tree mean rank: {tree_mean:.3f}" if np.isfinite(tree_mean) else "- Tree mean rank: unavailable",
            f"- TST mean rank: {tst_mean:.3f}" if np.isfinite(tst_mean) else "- TST mean rank: unavailable",
            f"- {interpretation}",
            "",
        ]
    )
    path.write_text(path.read_text() + "\n".join(lines) + "\n")


def _diagnostic_error_row(
    *,
    model: str,
    dataset: str,
    seed: int,
    task: str,
    variant: str,
    n_samples: int,
    n_features: int,
    error: Exception,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": seed,
        "task": task,
        "family": "ablation",
        "model": model,
        "variant": variant,
        "n_samples": n_samples,
        "n_features": n_features,
        "status": "error",
        "error_message": str(error),
        "epoch": "",
        "train_loss": "",
        "train_metric": "",
        "val_metric": "",
        "grad_norm": "",
        "grad_norm_max": "",
        "prediction_mean": "",
        "prediction_std": "",
        "train_val_gap": "",
        "has_gate": "",
        "gate_mean": "",
        "gate_median": "",
        "gate_sparsity": "",
        "best_epoch": "",
        "best_val_metric": "",
        "final_val_metric": "",
        "final_vs_best": "",
        "early_stopped": "",
        "effective_training_status": "unstable",
    }


def run_benchmark(
    suite: str = "synthetic",
    *,
    output_path: str | Path = "reports/benchmark_results.md",
    csv_output_path: str | Path | None = None,
    diagnostics_output_path: str | Path | None = None,
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
    diagnostic_rows: list[dict[str, object]] = []
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
                    for diagnostic in trained.diagnostics:
                        diagnostic_rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "task": bundle.task_type,
                                "family": "ablation",
                                "model": label,
                                "variant": label,
                                "n_samples": total_samples,
                                "n_features": raw_features,
                                "status": "ok",
                                "error_message": "",
                                **diagnostic,
                            }
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
                    diagnostic_rows.append(
                        _diagnostic_error_row(
                            model=label,
                            dataset=dataset_name,
                            seed=seed,
                            task=bundle.task_type,
                            variant=label,
                            n_samples=total_samples,
                            n_features=raw_features,
                            error=exc,
                        )
                    )

    write_markdown_table([result.as_row() for result in results], output_path)
    _append_benchmark_summary(results, output_path)
    if csv_output_path is not None:
        write_csv_table([result.as_row() for result in results], csv_output_path)
    if diagnostics_output_path is not None:
        write_csv_table(diagnostic_rows, diagnostics_output_path)
    return results
