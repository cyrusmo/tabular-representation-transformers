from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.evaluation.reporting import write_csv_table, write_markdown_table
from tabular_state_transformer.models.ft_transformer import FTTransformerStyle
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
    benchmark_mode: str = "default_benchmark"
    base_variant: str = ""
    selected_config_id: str = ""
    selected_lr: str = ""
    selection_metric: str = ""
    selection_mode: str = ""
    selection_score: str = ""
    selected_epoch: str = ""
    candidate_config_count: str = ""
    candidate_lrs: str = ""
    tuning_budget_type: str = ""

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
            "benchmark_mode": self.benchmark_mode,
            "base_variant": self.base_variant,
            "selected_config_id": self.selected_config_id,
            "selected_lr": self.selected_lr,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "selection_score": self.selection_score,
            "selected_epoch": self.selected_epoch,
            "candidate_config_count": self.candidate_config_count,
            "candidate_lrs": self.candidate_lrs,
            "tuning_budget_type": self.tuning_budget_type,
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
BENCHMARK_MODES = {"default_benchmark", "tuned_tst_benchmark"}
DEFAULT_TUNING_LRS = [1e-4, 3e-4, 1e-3]
DEFAULT_TUNING_LRS_TEXT = "1e-4,3e-4,1e-3"

MODEL_CONFIGS = {
    "TST-v0": "configs/model/tst_v0.yaml",
    "TST-v1-Gate": "configs/model/tst_v1_gate.yaml",
    "TST-v2-GateFourier": "configs/model/tst_v2_fourier_gate.yaml",
    "TST-v3-MoE": "configs/model/tst_v3_moe.yaml",
}

NEURAL_BASELINE_CONFIGS = {
    "ft_transformer": {
        "label": "FT-Transformer-style",
        "variant": "local_ft_transformer",
        "config": "configs/model/ft_transformer.yaml",
        "model_factory": FTTransformerStyle,
    }
}


@dataclass
class TuningCandidate:
    trained: object
    lr: float
    order: int
    config_id: str
    fit_seconds: float
    selection_metric: str
    selection_mode: str
    selection_score: float
    selected_epoch: int


def _metric(task: str, y_true: np.ndarray, pred) -> tuple[str, float]:
    if task == "classification":
        labels = np.asarray(pred).argmax(axis=1) if np.asarray(pred).ndim == 2 else pred
        return "accuracy", float(accuracy_score(y_true, labels))
    return "rmse", float(mean_squared_error(y_true, np.asarray(pred).reshape(-1)) ** 0.5)


def _selection_info(task: str) -> tuple[str, str]:
    if task == "classification":
        return "val_accuracy", "maximize"
    return "val_rmse", "minimize"


def _format_lr(value: float) -> str:
    text = f"{value:.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e")


def _safe_id(value: str) -> str:
    value = value.replace("MoE", "Moe")
    snake_cased = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_cased.lower())
    return normalized.strip("_")


def _selected_config_id(base_variant: str, lr: float) -> str:
    return f"{_safe_id(base_variant)}_lr_{_format_lr(lr)}"


def _candidate_sort_key(task: str, candidate: TuningCandidate) -> tuple[float, int, float, int]:
    score_key = -candidate.selection_score if task == "classification" else candidate.selection_score
    return (score_key, candidate.selected_epoch, candidate.lr, candidate.order)


def _select_best_tuning_candidate(task: str, candidates: list[TuningCandidate]) -> TuningCandidate:
    if not candidates:
        raise ValueError("No successful tuning candidates were available.")
    return min(candidates, key=lambda candidate: _candidate_sort_key(task, candidate))


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
    benchmark_mode: str = "default_benchmark",
    base_variant: str = "",
    selected_config_id: str = "",
    selected_lr: str = "",
    selection_metric: str = "",
    selection_mode: str = "",
    candidate_config_count: str = "",
    candidate_lrs: str = "",
    tuning_budget_type: str = "",
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
        benchmark_mode=benchmark_mode,
        base_variant=base_variant,
        selected_config_id=selected_config_id,
        selected_lr=selected_lr,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        candidate_config_count=candidate_config_count,
        candidate_lrs=candidate_lrs,
        tuning_budget_type=tuning_budget_type,
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
    by_model: dict[tuple[str, str], dict[str, object]] = {}
    for result, rank in ranked:
        stats = by_model.setdefault(
            (result.model, result.variant),
            {
                "model": result.model,
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
    for stats in by_model.values():
        ranks = stats["ranks"]
        assert isinstance(ranks, list)
        summary_rows.append(
            {
                "model": stats["model"],
                "family": stats["family"],
                "variant": stats["variant"],
                "mean_rank": sum(ranks) / len(ranks),
                "wins": stats["wins"],
                "ok_rows": stats["ok_rows"],
            }
        )
    summary_rows.sort(key=lambda row: (float(row["mean_rank"]), str(row["model"])))

    tree_ranks = [rank for result, rank in ranked if result.model in TREE_BASELINE_LABELS]
    untuned_tst_ranks = [
        rank for result, rank in ranked if result.family == "ablation" and not result.base_variant
    ]
    tuned_tst_ranks = [rank for result, rank in ranked if result.family == "ablation" and result.base_variant]
    ft_ranks = [rank for result, rank in ranked if result.variant == "local_ft_transformer"]
    tree_mean = sum(tree_ranks) / len(tree_ranks) if tree_ranks else float("nan")
    tst_mean = sum(untuned_tst_ranks) / len(untuned_tst_ranks) if untuned_tst_ranks else float("nan")
    tuned_tst_mean = sum(tuned_tst_ranks) / len(tuned_tst_ranks) if tuned_tst_ranks else float("nan")
    ft_mean = sum(ft_ranks) / len(ft_ranks) if ft_ranks else float("nan")
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
    if np.isfinite(tuned_tst_mean):
        improvement = tst_mean - tuned_tst_mean if np.isfinite(tst_mean) else float("nan")
        remaining_gap = tuned_tst_mean - tree_mean if np.isfinite(tree_mean) else float("nan")
        lines.extend(
            [
                "## Fair TST Tuning Summary",
                "",
                f"- Untuned TST mean rank: {tst_mean:.3f}" if np.isfinite(tst_mean) else "- Untuned TST mean rank: unavailable",
                f"- Tuned TST mean rank: {tuned_tst_mean:.3f}",
                f"- Tuning improvement: {improvement:.3f} mean-rank points" if np.isfinite(improvement) else "- Tuning improvement: unavailable",
                f"- Remaining gap to trees: {remaining_gap:.3f} mean-rank points" if np.isfinite(remaining_gap) else "- Remaining gap to trees: unavailable",
                "- Tuned rows use a fixed three-candidate learning-rate budget, not a broad hyperparameter search.",
                "",
            ]
        )
    if np.isfinite(ft_mean):
        lines.extend(
            [
                "## Neural Baseline Summary",
                "",
                f"- Local FT-Transformer-style mean rank: {ft_mean:.3f}",
                "- This is a local FT-Transformer-style baseline, not a validated reference-paper reproduction.",
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
    family: str,
    variant: str,
    n_samples: int,
    n_features: int,
    error: Exception,
    benchmark_mode: str,
    base_variant: str = "",
    selected_config_id: str = "",
    selected_lr: str = "",
    selection_metric: str = "",
    selection_mode: str = "",
    selection_score: str = "",
    selected_epoch: str = "",
    candidate_config_count: str = "",
    candidate_lrs: str = "",
    tuning_budget_type: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": seed,
        "task": task,
        "family": family,
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
        "benchmark_mode": benchmark_mode,
        "base_variant": base_variant,
        "selected_config_id": selected_config_id,
        "selected_lr": selected_lr,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_score": selection_score,
        "selected_epoch": selected_epoch,
        "candidate_config_count": candidate_config_count,
        "candidate_lrs": candidate_lrs,
        "tuning_budget_type": tuning_budget_type,
    }


def _append_training_diagnostics(
    diagnostic_rows: list[dict[str, object]],
    *,
    trained,
    dataset: str,
    seed: int,
    task: str,
    family: str,
    model: str,
    variant: str,
    n_samples: int,
    n_features: int,
    benchmark_mode: str,
    base_variant: str = "",
    selected_config_id: str = "",
    selected_lr: str = "",
    selection_metric: str = "",
    selection_mode: str = "",
    selection_score: str = "",
    selected_epoch: str = "",
    candidate_config_count: str = "",
    candidate_lrs: str = "",
    tuning_budget_type: str = "",
) -> None:
    for diagnostic in trained.diagnostics:
        diagnostic_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "task": task,
                "family": family,
                "model": model,
                "variant": variant,
                "n_samples": n_samples,
                "n_features": n_features,
                "status": "ok",
                "error_message": "",
                "benchmark_mode": benchmark_mode,
                "base_variant": base_variant,
                "selected_config_id": selected_config_id,
                "selected_lr": selected_lr,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_score": selection_score,
                "selected_epoch": selected_epoch,
                "candidate_config_count": candidate_config_count,
                "candidate_lrs": candidate_lrs,
                "tuning_budget_type": tuning_budget_type,
                **diagnostic,
            }
        )


def _make_config(
    repo_root: Path,
    config_path: str,
    *,
    task: str,
    max_epochs: int,
    seed: int,
    learning_rate: float | None = None,
) -> TabularStateConfig:
    values = read_yaml(repo_root / config_path)
    values.update(
        {
            "task": task,
            "max_epochs": max_epochs,
            "n_features": 1,
            "random_state": seed,
        }
    )
    if learning_rate is not None:
        values["learning_rate"] = learning_rate
    return TabularStateConfig.from_dict(values)


def _predict_trained(trained, bundle) -> tuple[float, np.ndarray]:
    X_test = trained.preprocessor.transform(bundle.X_test).astype("float32")
    start = perf_counter()
    pred = trained.model.predict_numpy(X_test)
    return perf_counter() - start, pred


def _selection_score_text(value: float) -> str:
    return f"{value:.6f}"


def run_benchmark(
    suite: str = "synthetic",
    *,
    output_path: str | Path = "reports/benchmark_results.md",
    csv_output_path: str | Path | None = None,
    diagnostics_output_path: str | Path | None = None,
    n_samples: int = 512,
    max_epochs: int = 2,
    tuning_max_epochs: int | None = None,
    benchmark_mode: str = "default_benchmark",
    seeds: Sequence[int] | None = None,
    dataset_names: Sequence[str] | None = None,
    baselines: Sequence[str] | None = None,
    model_configs: Sequence[str] | None = None,
    neural_baselines: Sequence[str] | None = None,
    continue_on_error: bool = True,
) -> list[BenchmarkResult]:
    if benchmark_mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode '{benchmark_mode}'.")
    selected_datasets = list(_datasets_for_suite(suite) if dataset_names is None else dataset_names)
    selected_seeds = list([42] if seeds is None else seeds)
    selected_baselines = list(DEFAULT_BASELINES if baselines is None else baselines)
    selected_models = list(MODEL_CONFIGS if model_configs is None else model_configs)
    selected_neural_baselines = list([] if neural_baselines is None else neural_baselines)
    resolved_tuning_max_epochs = tuning_max_epochs or max_epochs
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
                            benchmark_mode=benchmark_mode,
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
                            benchmark_mode=benchmark_mode,
                        )
                    )

            for label in selected_models:
                try:
                    config = _make_config(
                        repo_root,
                        MODEL_CONFIGS[label],
                        task=bundle.task_type,
                        max_epochs=max_epochs,
                        seed=seed,
                    )
                    start = perf_counter()
                    trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
                    fit_seconds = perf_counter() - start
                    predict_seconds, pred = _predict_trained(trained, bundle)
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
                            benchmark_mode=benchmark_mode,
                        )
                    )
                    _append_training_diagnostics(
                        diagnostic_rows,
                        trained=trained,
                        dataset=dataset_name,
                        seed=seed,
                        task=bundle.task_type,
                        family="ablation",
                        model=label,
                        variant=label,
                        n_samples=total_samples,
                        n_features=raw_features,
                        benchmark_mode=benchmark_mode,
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
                            benchmark_mode=benchmark_mode,
                        )
                    )
                    diagnostic_rows.append(
                        _diagnostic_error_row(
                            model=label,
                            dataset=dataset_name,
                            seed=seed,
                            task=bundle.task_type,
                            family="ablation",
                            variant=label,
                            n_samples=total_samples,
                            n_features=raw_features,
                            error=exc,
                            benchmark_mode=benchmark_mode,
                        )
                    )

            for neural_name in selected_neural_baselines:
                spec = NEURAL_BASELINE_CONFIGS[neural_name]
                label = str(spec["label"])
                variant = str(spec["variant"])
                try:
                    config = _make_config(
                        repo_root,
                        str(spec["config"]),
                        task=bundle.task_type,
                        max_epochs=max_epochs,
                        seed=seed,
                    )
                    start = perf_counter()
                    trained = Trainer(
                        config,
                        max_epochs=max_epochs,
                        batch_size=128,
                        model_factory=spec["model_factory"],
                    ).fit(bundle)
                    fit_seconds = perf_counter() - start
                    predict_seconds, pred = _predict_trained(trained, bundle)
                    metric, score = _metric(bundle.task_type, bundle.y_test, pred)
                    results.append(
                        BenchmarkResult(
                            label,
                            dataset_name,
                            seed,
                            bundle.task_type,
                            "neural_baseline",
                            variant,
                            metric,
                            score,
                            fit_seconds,
                            predict_seconds,
                            total_samples,
                            raw_features,
                            benchmark_mode=benchmark_mode,
                        )
                    )
                    _append_training_diagnostics(
                        diagnostic_rows,
                        trained=trained,
                        dataset=dataset_name,
                        seed=seed,
                        task=bundle.task_type,
                        family="neural_baseline",
                        model=label,
                        variant=variant,
                        n_samples=total_samples,
                        n_features=raw_features,
                        benchmark_mode=benchmark_mode,
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
                            family="neural_baseline",
                            variant=variant,
                            n_samples=total_samples,
                            n_features=raw_features,
                            error=exc,
                            benchmark_mode=benchmark_mode,
                        )
                    )
                    diagnostic_rows.append(
                        _diagnostic_error_row(
                            model=label,
                            dataset=dataset_name,
                            seed=seed,
                            task=bundle.task_type,
                            family="neural_baseline",
                            variant=variant,
                            n_samples=total_samples,
                            n_features=raw_features,
                            error=exc,
                            benchmark_mode=benchmark_mode,
                        )
                    )

            if benchmark_mode == "tuned_tst_benchmark":
                for label in selected_models:
                    selection_metric, selection_mode = _selection_info(bundle.task_type)
                    candidates: list[TuningCandidate] = []
                    candidate_errors: list[str] = []
                    for order, lr in enumerate(DEFAULT_TUNING_LRS):
                        config_id = _selected_config_id(label, lr)
                        try:
                            config = _make_config(
                                repo_root,
                                MODEL_CONFIGS[label],
                                task=bundle.task_type,
                                max_epochs=resolved_tuning_max_epochs,
                                seed=seed,
                                learning_rate=lr,
                            )
                            start = perf_counter()
                            trained = Trainer(
                                config,
                                lr=lr,
                                max_epochs=resolved_tuning_max_epochs,
                                batch_size=128,
                            ).fit(bundle)
                            fit_seconds = perf_counter() - start
                            candidates.append(
                                TuningCandidate(
                                    trained=trained,
                                    lr=lr,
                                    order=order,
                                    config_id=config_id,
                                    fit_seconds=fit_seconds,
                                    selection_metric=selection_metric,
                                    selection_mode=selection_mode,
                                    selection_score=trained.best_val_metric,
                                    selected_epoch=trained.best_epoch,
                                )
                            )
                        except Exception as exc:
                            if not continue_on_error:
                                raise
                            candidate_errors.append(f"{config_id}: {exc}")
                    try:
                        selected = _select_best_tuning_candidate(bundle.task_type, candidates)
                        trained = selected.trained
                        fit_seconds = sum(candidate.fit_seconds for candidate in candidates)
                        predict_seconds, pred = _predict_trained(trained, bundle)
                        metric, score = _metric(bundle.task_type, bundle.y_test, pred)
                        selected_lr_text = _format_lr(selected.lr)
                        selected_score_text = _selection_score_text(selected.selection_score)
                        notes = "; ".join(candidate_errors)
                        results.append(
                            BenchmarkResult(
                                "TST",
                                dataset_name,
                                seed,
                                bundle.task_type,
                                "ablation",
                                f"{label}-tuned",
                                metric,
                                score,
                                fit_seconds,
                                predict_seconds,
                                total_samples,
                                raw_features,
                                notes=notes,
                                benchmark_mode=benchmark_mode,
                                base_variant=label,
                                selected_config_id=selected.config_id,
                                selected_lr=selected_lr_text,
                                selection_metric=selection_metric,
                                selection_mode=selection_mode,
                                selection_score=selected_score_text,
                                selected_epoch=str(selected.selected_epoch),
                                candidate_config_count=str(len(DEFAULT_TUNING_LRS)),
                                candidate_lrs=DEFAULT_TUNING_LRS_TEXT,
                                tuning_budget_type="lr_only",
                            )
                        )
                        _append_training_diagnostics(
                            diagnostic_rows,
                            trained=trained,
                            dataset=dataset_name,
                            seed=seed,
                            task=bundle.task_type,
                            family="ablation",
                            model="TST",
                            variant=f"{label}-tuned",
                            n_samples=total_samples,
                            n_features=raw_features,
                            benchmark_mode=benchmark_mode,
                            base_variant=label,
                            selected_config_id=selected.config_id,
                            selected_lr=selected_lr_text,
                            selection_metric=selection_metric,
                            selection_mode=selection_mode,
                            selection_score=selected_score_text,
                            selected_epoch=str(selected.selected_epoch),
                            candidate_config_count=str(len(DEFAULT_TUNING_LRS)),
                            candidate_lrs=DEFAULT_TUNING_LRS_TEXT,
                            tuning_budget_type="lr_only",
                        )
                    except Exception as exc:
                        if not continue_on_error:
                            raise
                        error = RuntimeError("; ".join(candidate_errors) or str(exc))
                        results.append(
                            _error_result(
                                model="TST",
                                dataset=dataset_name,
                                seed=seed,
                                task=bundle.task_type,
                                family="ablation",
                                variant=f"{label}-tuned",
                                n_samples=total_samples,
                                n_features=raw_features,
                                error=error,
                                benchmark_mode=benchmark_mode,
                                base_variant=label,
                                selection_metric=selection_metric,
                                selection_mode=selection_mode,
                                candidate_config_count=str(len(DEFAULT_TUNING_LRS)),
                                candidate_lrs=DEFAULT_TUNING_LRS_TEXT,
                                tuning_budget_type="lr_only",
                            )
                        )
                        diagnostic_rows.append(
                            _diagnostic_error_row(
                                model="TST",
                                dataset=dataset_name,
                                seed=seed,
                                task=bundle.task_type,
                                family="ablation",
                                variant=f"{label}-tuned",
                                n_samples=total_samples,
                                n_features=raw_features,
                                error=error,
                                benchmark_mode=benchmark_mode,
                                base_variant=label,
                                selection_metric=selection_metric,
                                selection_mode=selection_mode,
                                candidate_config_count=str(len(DEFAULT_TUNING_LRS)),
                                candidate_lrs=DEFAULT_TUNING_LRS_TEXT,
                                tuning_budget_type="lr_only",
                            )
                        )

    write_markdown_table([result.as_row() for result in results], output_path)
    _append_benchmark_summary(results, output_path)
    if csv_output_path is not None:
        write_csv_table([result.as_row() for result in results], csv_output_path)
    if diagnostics_output_path is not None:
        write_csv_table(diagnostic_rows, diagnostics_output_path)
    return results
