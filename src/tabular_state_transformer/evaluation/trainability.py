from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.data.schema import TabularDatasetBundle
from tabular_state_transformer.evaluation.benchmark import MODEL_CONFIGS
from tabular_state_transformer.evaluation.reporting import write_csv_table, write_markdown_table
from tabular_state_transformer.models.baselines import make_baseline
from tabular_state_transformer.training import Trainer
from tabular_state_transformer.utils.io import read_yaml


@dataclass(frozen=True)
class TrainabilityTask:
    label: str
    dataset_name: str
    kwargs: dict[str, object]


AUDIT_TASKS = {
    "xor_2f": TrainabilityTask(
        "synthetic_xor_2f",
        "synthetic_xor",
        {"n_features": 2, "noise": 0.0},
    ),
    "xor_20f": TrainabilityTask(
        "synthetic_xor_20f",
        "synthetic_xor",
        {"n_features": 20, "noise": 0.0},
    ),
    "irrelevant_noise_100f": TrainabilityTask(
        "synthetic_irrelevant_noise_100f",
        "synthetic_irrelevant_noise",
        {"n_features": 100},
    ),
}

DEFAULT_AUDIT_TASKS = ["xor_2f", "xor_20f", "irrelevant_noise_100f"]
DEFAULT_AUDIT_MODELS = ["mlp", "TST-v0", "TST-v1-Gate", "TST-v4-CLS", "TST-v5-CLS-Cross"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_task(task_name: str, *, seed: int, n_samples: int) -> tuple[str, TabularDatasetBundle]:
    try:
        task = AUDIT_TASKS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown trainability task '{task_name}'.") from exc
    bundle = load_dataset(
        task.dataset_name,
        split_seed=seed,
        n_samples=n_samples,
        **task.kwargs,
    )
    return task.label, bundle


def _make_tst_config(
    model_name: str,
    *,
    task: str,
    seed: int,
    max_epochs: int,
) -> TabularStateConfig:
    try:
        config_path = MODEL_CONFIGS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown TST model '{model_name}'.") from exc
    values = read_yaml(_repo_root() / config_path)
    values.update(
        {
            "dropout": 0.0,
            "early_stopping_patience": None,
            "max_epochs": max_epochs,
            "n_features": 1,
            "random_state": seed,
            "task": task,
        }
    )
    return TabularStateConfig.from_dict(values)


def _encoded_targets(trained, y: np.ndarray) -> np.ndarray:
    if trained.class_labels is None:
        return y
    return np.searchsorted(trained.class_labels, y)


def _class_balance(labels: np.ndarray) -> str:
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return ";".join(f"{value}:{count / total:.3f}" for value, count in zip(values, counts, strict=True))


def _binary_logit_std_from_probability(probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    if clipped.ndim == 1:
        return float(np.std(np.log(clipped / (1.0 - clipped))))
    if clipped.shape[1] == 2:
        positive = clipped[:, 1]
        return float(np.std(np.log(positive / (1.0 - positive))))
    return float(np.std(np.log(clipped)))


def _tst_metrics(trained, bundle: TabularDatasetBundle) -> dict[str, object]:
    model = trained.model
    device = next(model.parameters()).device
    splits = {
        "train": (bundle.X_train, _encoded_targets(trained, bundle.y_train)),
        "val": (bundle.X_val, _encoded_targets(trained, bundle.y_val)),
        "test": (bundle.X_test, _encoded_targets(trained, bundle.y_test)),
    }
    metrics: dict[str, object] = {}
    test_logits: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    with torch.no_grad():
        for split, (frame, target) in splits.items():
            x = trained.preprocessor.transform(frame).astype("float32")
            output = model(torch.as_tensor(x, dtype=torch.float32, device=device))
            logits = output.detach().cpu().numpy()
            pred = logits.argmax(axis=1)
            metrics[f"restored_{split}_accuracy"] = float(accuracy_score(target, pred))
            if split == "test":
                test_logits = logits
                test_pred = pred
    diagnostics = trained.diagnostics
    max_train_accuracy = max(float(row["train_metric"]) for row in diagnostics) if diagnostics else float("nan")
    final_epoch_train_accuracy = float(diagnostics[-1]["train_metric"]) if diagnostics else float("nan")
    final_train_loss = float(diagnostics[-1]["train_loss"]) if diagnostics else float("nan")
    metrics.update(
        {
            "train_accuracy": max_train_accuracy,
            "final_epoch_train_accuracy": final_epoch_train_accuracy,
            "val_accuracy": trained.best_val_metric,
            "test_accuracy": metrics["restored_test_accuracy"],
            "train_loss": final_train_loss,
            "logit_std": float(np.std(test_logits)) if test_logits is not None else float("nan"),
            "prediction_class_balance": _class_balance(test_pred) if test_pred is not None else "",
            "best_epoch": trained.best_epoch,
            "early_stopped": trained.early_stopped,
            "effective_training_status": trained.effective_training_status,
        }
    )
    return metrics


def _run_tst_model(
    *,
    model_name: str,
    dataset_label: str,
    bundle: TabularDatasetBundle,
    seed: int,
    n_samples: int,
    max_epochs: int,
    batch_size: int,
    device: str,
    diagnostic_rows: list[dict[str, object]],
) -> dict[str, object]:
    config = _make_tst_config(model_name, task=bundle.task_type, seed=seed, max_epochs=max_epochs)
    start = perf_counter()
    trained = Trainer(
        config,
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping_patience=0,
        device=device,
    ).fit(bundle)
    fit_seconds = perf_counter() - start
    row = {
        "model": model_name,
        "dataset": dataset_label,
        "seed": seed,
        "family": "ablation",
        "status": "ok",
        "error_message": "",
        "n_samples": n_samples,
        "n_features": bundle.X_train.shape[1],
        "max_epochs": max_epochs,
        "dropout": 0.0,
        "early_stopping": "disabled",
        "fit_seconds": fit_seconds,
    }
    row.update(_tst_metrics(trained, bundle))
    for diagnostic in trained.diagnostics:
        diagnostic_rows.append(
            {
                "dataset": dataset_label,
                "seed": seed,
                "family": "ablation",
                "model": model_name,
                "status": "ok",
                "error_message": "",
                "n_samples": n_samples,
                "n_features": bundle.X_train.shape[1],
                **diagnostic,
            }
        )
    return row


def _run_mlp_model(
    *,
    dataset_label: str,
    bundle: TabularDatasetBundle,
    seed: int,
    n_samples: int,
    max_epochs: int,
) -> dict[str, object]:
    start = perf_counter()
    estimator = make_baseline("mlp", bundle.task_type, bundle.X_train, random_state=seed)
    estimator.set_params(model__max_iter=max_epochs)
    estimator.fit(bundle.X_train, bundle.y_train)
    fit_seconds = perf_counter() - start
    train_pred = estimator.predict(bundle.X_train)
    val_pred = estimator.predict(bundle.X_val)
    test_pred = estimator.predict(bundle.X_test)
    probabilities = estimator.predict_proba(bundle.X_test)
    mlp = estimator.named_steps["model"]
    return {
        "model": "MLP",
        "dataset": dataset_label,
        "seed": seed,
        "family": "baseline",
        "status": "ok",
        "error_message": "",
        "n_samples": n_samples,
        "n_features": bundle.X_train.shape[1],
        "max_epochs": max_epochs,
        "dropout": "",
        "early_stopping": "disabled",
        "fit_seconds": fit_seconds,
        "train_accuracy": float(accuracy_score(bundle.y_train, train_pred)),
        "final_epoch_train_accuracy": float(accuracy_score(bundle.y_train, train_pred)),
        "restored_train_accuracy": float(accuracy_score(bundle.y_train, train_pred)),
        "val_accuracy": float(accuracy_score(bundle.y_val, val_pred)),
        "restored_val_accuracy": float(accuracy_score(bundle.y_val, val_pred)),
        "test_accuracy": float(accuracy_score(bundle.y_test, test_pred)),
        "restored_test_accuracy": float(accuracy_score(bundle.y_test, test_pred)),
        "train_loss": float(getattr(mlp, "loss_", np.nan)),
        "logit_std": _binary_logit_std_from_probability(probabilities),
        "prediction_class_balance": _class_balance(np.asarray(test_pred)),
        "best_epoch": int(getattr(mlp, "n_iter_", max_epochs)),
        "early_stopped": False,
        "effective_training_status": "",
    }


def _error_row(
    *,
    model_name: str,
    dataset_label: str,
    seed: int,
    n_samples: int,
    n_features: int,
    max_epochs: int,
    error: Exception,
) -> dict[str, object]:
    return {
        "model": "MLP" if model_name == "mlp" else model_name,
        "dataset": dataset_label,
        "seed": seed,
        "family": "baseline" if model_name == "mlp" else "ablation",
        "status": "error",
        "error_message": str(error),
        "n_samples": n_samples,
        "n_features": n_features,
        "max_epochs": max_epochs,
        "train_accuracy": float("nan"),
        "val_accuracy": float("nan"),
        "test_accuracy": float("nan"),
        "train_loss": float("nan"),
        "logit_std": float("nan"),
        "prediction_class_balance": "",
        "best_epoch": "",
    }


def _append_summary(rows: list[dict[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    xor_2f_tst = [
        row
        for row in ok_rows
        if row["dataset"] == "synthetic_xor_2f" and row["family"] == "ablation"
    ]
    xor_20f_tst = [
        row
        for row in ok_rows
        if row["dataset"] == "synthetic_xor_20f" and row["family"] == "ablation"
    ]
    noise_tst = [
        row
        for row in ok_rows
        if row["dataset"] == "synthetic_irrelevant_noise_100f" and row["family"] == "ablation"
    ]
    best_xor_2f_train = max((float(row["train_accuracy"]) for row in xor_2f_tst), default=float("nan"))
    best_xor_20f_train = max((float(row["train_accuracy"]) for row in xor_20f_tst), default=float("nan"))
    best_noise_train = max((float(row["train_accuracy"]) for row in noise_tst), default=float("nan"))
    if np.isfinite(best_xor_2f_train) and best_xor_2f_train < 0.95:
        verdict = "TST failed the 2-feature XOR memorization check; prioritize optimizer/model trainability debugging."
    elif np.isfinite(best_xor_20f_train) and best_xor_20f_train < 0.95:
        verdict = "TST can handle the easiest setting better than the high-dimensional XOR setting; prioritize feature/noise handling."
    elif np.isfinite(best_noise_train) and best_noise_train < 0.95:
        verdict = "TST can memorize XOR but not the irrelevant-noise task; prioritize feature selection and gate behavior."
    else:
        verdict = "TST can memorize the audit tasks; prioritize generalization and regularization diagnostics."
    lines = [
        "",
        "## Trainability Verdict",
        "",
        f"- Best TST train accuracy on 2-feature XOR: {best_xor_2f_train:.3f}" if np.isfinite(best_xor_2f_train) else "- Best TST train accuracy on 2-feature XOR: unavailable",
        f"- Best TST train accuracy on 20-feature XOR: {best_xor_20f_train:.3f}" if np.isfinite(best_xor_20f_train) else "- Best TST train accuracy on 20-feature XOR: unavailable",
        f"- Best TST train accuracy on irrelevant-noise task: {best_noise_train:.3f}" if np.isfinite(best_noise_train) else "- Best TST train accuracy on irrelevant-noise task: unavailable",
        f"- {verdict}",
        "",
    ]
    path.write_text(path.read_text() + "\n".join(lines))


def run_trainability_audit(
    *,
    output_path: str | Path = "reports/trainability_audit_results.md",
    csv_output_path: str | Path | None = "reports/trainability_audit_results.csv",
    diagnostics_output_path: str | Path | None = "reports/trainability_audit_diagnostics.csv",
    task_names: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_samples: int = 512,
    max_epochs: int = 300,
    batch_size: int = 128,
    device: str = "cpu",
    continue_on_error: bool = True,
) -> list[dict[str, object]]:
    selected_tasks = list(DEFAULT_AUDIT_TASKS if task_names is None else task_names)
    selected_models = list(DEFAULT_AUDIT_MODELS if model_names is None else model_names)
    selected_seeds = list([42] if seeds is None else seeds)
    results: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for seed in selected_seeds:
        for task_name in selected_tasks:
            dataset_label, bundle = _load_task(task_name, seed=seed, n_samples=n_samples)
            for model_name in selected_models:
                try:
                    if model_name == "mlp":
                        results.append(
                            _run_mlp_model(
                                dataset_label=dataset_label,
                                bundle=bundle,
                                seed=seed,
                                n_samples=n_samples,
                                max_epochs=max_epochs,
                            )
                        )
                    else:
                        results.append(
                            _run_tst_model(
                                model_name=model_name,
                                dataset_label=dataset_label,
                                bundle=bundle,
                                seed=seed,
                                n_samples=n_samples,
                                max_epochs=max_epochs,
                                batch_size=batch_size,
                                device=device,
                                diagnostic_rows=diagnostic_rows,
                            )
                        )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    results.append(
                        _error_row(
                            model_name=model_name,
                            dataset_label=dataset_label,
                            seed=seed,
                            n_samples=n_samples,
                            n_features=bundle.X_train.shape[1],
                            max_epochs=max_epochs,
                            error=exc,
                        )
                    )
    write_markdown_table(results, output_path)
    _append_summary(results, output_path)
    if csv_output_path is not None:
        write_csv_table(results, csv_output_path)
    if diagnostics_output_path is not None:
        write_csv_table(diagnostic_rows, diagnostics_output_path)
    return results
