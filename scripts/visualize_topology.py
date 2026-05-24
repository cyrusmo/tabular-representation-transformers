from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.evaluation.benchmark import MODEL_CONFIGS
from tabular_state_transformer.models.baselines import make_baseline
from tabular_state_transformer.training import Trainer
from tabular_state_transformer.utils.io import read_yaml


TRUE_2D_SURFACE_TASKS = [
    "synthetic_axis_aligned",
    "synthetic_xor",
    "synthetic_rotated",
    "synthetic_piecewise",
]

TASK_ALIASES = {
    "synthetic_axis_threshold": "synthetic_axis_aligned",
    "synthetic_threshold": "synthetic_axis_aligned",
    "synthetic_thresholds": "synthetic_axis_aligned",
}

SURFACE_MODELS = [
    ("random_forest", "RandomForest"),
    ("mlp", "MLP"),
    ("TST-v0", "TST-v0"),
    ("TST-v1-Gate", "TST-v1"),
    ("TST-v2-GateFourier", "TST-v2"),
]

GATE_MODELS = ["TST-v1-Gate", "TST-v2-GateFourier"]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_tasks(value: str) -> list[str]:
    tasks = []
    for item in value.split(","):
        task = item.strip()
        if not task:
            continue
        tasks.append(TASK_ALIASES.get(task, task))
    return tasks


def _load_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    return pd.read_csv(path).astype(str).to_dict("records")


def _draw_boxes(
    steps: list[str],
    output_path: Path,
    *,
    title: str,
    figsize: tuple[float, float] = (7.5, 8.0),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    y_positions = np.linspace(0.92, 0.08, len(steps))
    for i, (step, y) in enumerate(zip(steps, y_positions, strict=True)):
        ax.text(
            0.5,
            y,
            step,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4f7fb", "edgecolor": "#53687e"},
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.035),
                xytext=(0.5, y - 0.035),
                arrowprops={"arrowstyle": "->", "color": "#34495e", "lw": 1.5},
            )
    ax.set_title(title, fontsize=14, pad=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_tensor_topology_schematic(output_dir: Path) -> Path:
    output_path = output_dir / "tensor_topology_schematic.png"
    _draw_boxes(
        [
            "Raw tabular row",
            "Feature tokens: [x1] [x2] [x3] ... [xF]",
            "Column identity embeddings",
            "Optional sparse feature gate",
            "Optional Fourier expansion",
            "Feature interaction block",
            "Pooled state",
            "Prediction head",
        ],
        output_path,
        title="Tabular State Transformer Topology",
    )
    return output_path


def make_representation_shape_diagram(output_dir: Path) -> Path:
    output_path = output_dir / "representation_shape_diagram.png"
    _draw_boxes(
        [
            "Batch x Features",
            "Tokenizer",
            "Batch x Features x d_model",
            "Interaction block",
            "Batch x Features x d_model",
            "Feature pooling",
            "Batch x d_model",
        ],
        output_path,
        title="Representation Shape Flow",
        figsize=(7.0, 7.0),
    )
    return output_path


def _make_tst_config(label: str, task: str, seed: int, max_epochs: int) -> TabularStateConfig:
    values = read_yaml(Path(MODEL_CONFIGS[label]))
    values.update({"task": task, "random_state": seed, "max_epochs": max_epochs, "n_features": 1})
    return TabularStateConfig.from_dict(values)


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _mean_gate_activations(values) -> np.ndarray:
    array = _as_numpy(values)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.ndim == 2:
        return array.mean(axis=0).reshape(-1)
    return array.mean(axis=tuple(i for i in range(array.ndim) if i != 1)).reshape(-1)


def extract_gate_values(model, X_valid: np.ndarray | None = None) -> np.ndarray | None:
    if hasattr(model, "get_feature_gate_values"):
        values = model.get_feature_gate_values(X_valid)
        return None if values is None else _as_numpy(values).reshape(-1)

    gate = getattr(model, "gate", None)
    if gate is None:
        return None

    for method_name in ("get_feature_gate_values", "gate_values", "compute_gate_values"):
        if hasattr(gate, method_name):
            method = getattr(gate, method_name)
            values = method(X_valid) if X_valid is not None else method()
            return None if values is None else _mean_gate_activations(values)

    if X_valid is not None and hasattr(gate, "last_gate_values"):
        with torch.no_grad():
            model(torch.as_tensor(X_valid, dtype=torch.float32))
        values = getattr(gate, "last_gate_values")
        return None if values is None else _mean_gate_activations(values)

    if hasattr(gate, "logits"):
        return torch.sigmoid(gate.logits).detach().cpu().numpy().reshape(-1)

    return None


def _feature_vector(values: np.ndarray | None, n_features: int) -> np.ndarray:
    if values is None:
        return np.full(n_features, np.nan)
    values = np.asarray(values, dtype="float32").reshape(-1)
    if len(values) == n_features:
        return values
    output = np.full(n_features, np.nan, dtype="float32")
    output[: min(n_features, len(values))] = values[:n_features]
    return output


def make_feature_gate_heatmap(
    output_dir: Path,
    *,
    seeds: list[int],
    n_samples: int,
    max_epochs: int,
    n_features: int,
) -> Path:
    output_path = output_dir / "feature_gate_heatmap.png"
    rows: list[str] = []
    values: list[np.ndarray] = []
    datasets = ["synthetic_irrelevant_noise", "synthetic_sparse_high_order"]

    for dataset_name in datasets:
        for seed in seeds:
            bundle = load_dataset(
                dataset_name,
                split_seed=seed,
                n_samples=n_samples,
                n_features=n_features,
            )
            for label in GATE_MODELS:
                config = _make_tst_config(label, bundle.task_type, seed, max_epochs)
                trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
                X_val = trained.preprocessor.transform(bundle.X_val).astype("float32")
                gate_values = extract_gate_values(trained.model, X_val)
                values.append(_feature_vector(gate_values, n_features))
                rows.append(f"{dataset_name}\nseed={seed}\n{label}")

    matrix = np.vstack(values)
    fig, ax = plt.subplots(figsize=(max(8.0, n_features * 0.35), max(4.0, len(rows) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Learned Sparse Feature Gate Values")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Dataset / Seed / Variant")
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels([f"x{i}" for i in range(n_features)], rotation=90)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    fig.colorbar(im, ax=ax, label="sigmoid(gate logit)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _fit_surface_model(model_key: str, bundle, *, seed: int, max_epochs: int):
    if model_key in {"random_forest", "mlp"}:
        model = make_baseline(model_key, bundle.task_type, bundle.X_train, random_state=seed)
        model.fit(bundle.X_train, bundle.y_train)
        return model, None

    config = _make_tst_config(model_key, bundle.task_type, seed, max_epochs)
    trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
    return trained.model, trained.preprocessor


def _predict_surface(model, preprocessor, frame: pd.DataFrame, task: str) -> np.ndarray:
    if preprocessor is None:
        if task == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(frame)[:, 1]
        return np.asarray(model.predict(frame)).reshape(-1)
    X = preprocessor.transform(frame).astype("float32")
    pred = model.predict_numpy(X)
    if task == "classification" and pred.ndim == 2:
        return pred[:, 1]
    return np.asarray(pred).reshape(-1)


def make_synthetic_task_topology_grid(
    output_dir: Path,
    *,
    tasks: list[str],
    seed: int,
    n_samples: int,
    max_epochs: int,
    grid_size: int,
) -> Path:
    non_2d_tasks = [task for task in tasks if task not in TRUE_2D_SURFACE_TASKS]
    if non_2d_tasks:
        raise ValueError(f"Decision surfaces require true 2D tasks. Unsupported: {non_2d_tasks}")

    output_path = output_dir / "synthetic_task_topology_grid.png"
    fig, axes = plt.subplots(
        len(tasks),
        len(SURFACE_MODELS),
        figsize=(len(SURFACE_MODELS) * 3.0, len(tasks) * 2.8),
        squeeze=False,
    )

    for row, dataset_name in enumerate(tasks):
        bundle = load_dataset(dataset_name, split_seed=seed, n_samples=n_samples, n_features=2)
        x_min, x_max = bundle.X_train["x0"].min(), bundle.X_train["x0"].max()
        y_min, y_max = bundle.X_train["x1"].min(), bundle.X_train["x1"].max()
        margin_x = max((x_max - x_min) * 0.08, 0.05)
        margin_y = max((y_max - y_min) * 0.08, 0.05)
        xx, yy = np.meshgrid(
            np.linspace(x_min - margin_x, x_max + margin_x, grid_size),
            np.linspace(y_min - margin_y, y_max + margin_y, grid_size),
        )
        grid = pd.DataFrame({"x0": xx.ravel(), "x1": yy.ravel()})

        for col, (model_key, title) in enumerate(SURFACE_MODELS):
            ax = axes[row][col]
            model, preprocessor = _fit_surface_model(
                model_key,
                bundle,
                seed=seed,
                max_epochs=max_epochs,
            )
            zz = _predict_surface(model, preprocessor, grid, bundle.task_type).reshape(xx.shape)
            contour = ax.contourf(xx, yy, zz, levels=20, cmap="viridis", alpha=0.9)
            ax.scatter(
                bundle.X_train["x0"],
                bundle.X_train["x1"],
                c=bundle.y_train,
                cmap="coolwarm",
                s=7,
                alpha=0.45,
                edgecolors="none",
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(dataset_name.replace("synthetic_", ""))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == len(SURFACE_MODELS) - 1:
                fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Decision Surfaces On True 2D Synthetic Tasks", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_tensor_topology_doc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Tensor Topology

These figures make the Tabular State Transformer representation legible. They are research
artifacts, not claims of model superiority.

## Figures

- `reports/figures/tensor_topology_schematic.png`: raw row to feature tokens, gates, spectral expansion, interaction block, pooled state, and head.
- `reports/figures/representation_shape_diagram.png`: tensor shape flow from `Batch x Features` to `Batch x d_model`.
- `reports/figures/feature_gate_heatmap.png`: learned sparse-gate values for TST-v1/TST-v2 on synthetic stress tasks.
- `reports/figures/synthetic_task_topology_grid.png`: decision surfaces for true 2D synthetic tasks only.

## Generation

```bash
venv/bin/python scripts/visualize_topology.py \
  --results-csv reports/experiments/legacy/results.csv \
  --output-dir reports/figures \
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \
  --seeds 42,43,44 \
  --n-samples 1024 \
  --max-epochs 20
```

Gate values are extracted through `extract_gate_values(model, X_valid)`: global gates use
`sigmoid(logits)`, input-dependent gates can expose or cache validation-set activations, and missing
gates are represented as `NaN`.

Decision surfaces are intentionally limited to native two-feature tasks. Higher-dimensional tasks are
not projected into 2D here, because that would make the visualization look more faithful than it is.
""",
    )


def _update_benchmark_links(report_path: Path) -> None:
    if not report_path.exists():
        return
    marker_start = "<!-- topology-figures:start -->"
    marker_end = "<!-- topology-figures:end -->"
    section = f"""
{marker_start}

## Tensor Topology Figures

- ![Tensor topology](figures/tensor_topology_schematic.png)
- ![Representation shape flow](figures/representation_shape_diagram.png)
- ![Feature gate heatmap](figures/feature_gate_heatmap.png)
- ![Synthetic task topology grid](figures/synthetic_task_topology_grid.png)

{marker_end}
"""
    text = report_path.read_text()
    if marker_start in text and marker_end in text:
        prefix = text.split(marker_start)[0].rstrip()
        suffix = text.split(marker_end, 1)[1].lstrip()
        report_path.write_text(prefix + "\n\n" + section.strip() + "\n\n" + suffix)
    else:
        report_path.write_text(text.rstrip() + "\n\n" + section.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        default="reports/experiments/legacy/results.csv",
    )
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--docs-output", default="docs/tensor_topology.md")
    parser.add_argument(
        "--benchmark-report",
        default="reports/experiments/legacy/results.md",
    )
    parser.add_argument(
        "--tasks",
        default="synthetic_axis_aligned,synthetic_xor,synthetic_rotated,synthetic_piecewise",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--surface-seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--gate-n-features", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=70)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_ints(args.seeds)
    tasks = _parse_tasks(args.tasks)
    result_rows = _load_result_rows(Path(args.results_csv))

    paths = [
        make_tensor_topology_schematic(output_dir),
        make_representation_shape_diagram(output_dir),
        make_feature_gate_heatmap(
            output_dir,
            seeds=seeds,
            n_samples=args.n_samples,
            max_epochs=args.max_epochs,
            n_features=args.gate_n_features,
        ),
        make_synthetic_task_topology_grid(
            output_dir,
            tasks=tasks,
            seed=args.surface_seed,
            n_samples=args.n_samples,
            max_epochs=args.max_epochs,
            grid_size=args.grid_size,
        ),
    ]
    _write_tensor_topology_doc(Path(args.docs_output))
    _update_benchmark_links(Path(args.benchmark_report))
    print(
        {
            "figures": [str(path) for path in paths],
            "docs": args.docs_output,
            "results_rows": len(result_rows),
            "surface_tasks": tasks,
        }
    )


if __name__ == "__main__":
    main()
