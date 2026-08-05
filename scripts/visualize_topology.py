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
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.evaluation.benchmark import MODEL_CONFIGS
from tabular_state_transformer.models.baselines import make_baseline
from tabular_state_transformer.models.ft_transformer import FTTransformerStyle
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

FT_TRANSFORMER_CONFIG = "configs/model/ft_transformer.yaml"

SKLEARN_SURFACE_MODELS = {"linear", "random_forest", "mlp", "gradient_boosting", "lightgbm"}

# Column order for the 3D decision-landscape grid: simplest bias first, TST ablation ladder last.
SURFACE_MODELS_3D = [
    ("linear", "Linear"),
    ("random_forest", "RandomForest"),
    ("lightgbm", "LightGBM"),
    ("mlp", "MLP"),
    ("ft_transformer", "FT-Transformer"),
    ("TST-v0", "TST-v0"),
    ("TST-v1-Gate", "TST-v1"),
    ("TST-v2-GateFourier", "TST-v2"),
    ("TST-v3-MoE", "TST-v3"),
]

# Shared camera so bias differences read across panels, tasks, and future Track A figures.
LANDSCAPE_CAMERA = {"elev": 28, "azim": -60}

# Known signal features for synthetic_irrelevant_noise (see data/synthetic.py).
IRRELEVANT_NOISE_SIGNAL_FEATURES = [0, 1, 2]


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
    if model_key in SKLEARN_SURFACE_MODELS:
        model = make_baseline(model_key, bundle.task_type, bundle.X_train, random_state=seed)
        model.fit(bundle.X_train, bundle.y_train)
        return model, None

    if model_key == "ft_transformer":
        values = read_yaml(Path(FT_TRANSFORMER_CONFIG))
        values.update(
            {"task": bundle.task_type, "random_state": seed, "max_epochs": max_epochs, "n_features": 1}
        )
        config = TabularStateConfig.from_dict(values)
        trained = Trainer(
            config, max_epochs=max_epochs, batch_size=128, model_factory=FTTransformerStyle
        ).fit(bundle)
        return trained.model, trained.preprocessor

    config = _make_tst_config(model_key, bundle.task_type, seed, max_epochs)
    trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
    return trained.model, trained.preprocessor


def _resolve_3d_models() -> list[tuple[str, str]]:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return [
            ("gradient_boosting", "GBDT") if key == "lightgbm" else (key, label)
            for key, label in SURFACE_MODELS_3D
        ]
    return list(SURFACE_MODELS_3D)


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


def _safe_name(label: str) -> str:
    return label.replace("/", "-").replace(" ", "_")


def _compute_landscape_meshes(
    output_dir: Path,
    *,
    tasks: list[str],
    models: list[tuple[str, str]],
    seed: int,
    n_samples: int,
    max_epochs: int,
    grid_size: int,
) -> dict[tuple[str, str], dict]:
    """Fit every (task, model) pair once, score a shared mesh, and persist .npz artifacts.

    The .npz exports let a future Neural ODE/SDE fork re-render these landscapes
    without re-fitting Track B models.
    """
    non_2d_tasks = [task for task in tasks if task not in TRUE_2D_SURFACE_TASKS]
    if non_2d_tasks:
        raise ValueError(f"Decision landscapes require true 2D tasks. Unsupported: {non_2d_tasks}")

    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    meshes: dict[tuple[str, str], dict] = {}

    for dataset_name in tasks:
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

        for model_key, label in models:
            model, preprocessor = _fit_surface_model(
                model_key, bundle, seed=seed, max_epochs=max_epochs
            )
            zz = _predict_surface(model, preprocessor, grid, bundle.task_type).reshape(xx.shape)
            entry = {
                "xx": xx,
                "yy": yy,
                "zz": zz,
                "x_train": bundle.X_train[["x0", "x1"]].to_numpy(dtype="float32"),
                "y_train": np.asarray(bundle.y_train, dtype="float32"),
                "task_type": bundle.task_type,
            }
            meshes[(dataset_name, label)] = entry
            np.savez_compressed(
                mesh_dir / f"{dataset_name}__{_safe_name(label)}.npz",
                xx=xx,
                yy=yy,
                zz=zz,
                x_train=entry["x_train"],
                y_train=entry["y_train"],
                task_type=np.array(bundle.task_type),
            )
    return meshes


def _draw_landscape_panel(ax, entry: dict, *, z_min: float, z_max: float) -> None:
    ax.plot_surface(
        entry["xx"],
        entry["yy"],
        entry["zz"],
        cmap="viridis",
        vmin=z_min,
        vmax=z_max,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    points = entry["x_train"]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        np.full(points.shape[0], z_min),
        c=entry["y_train"],
        cmap="coolwarm",
        s=3,
        alpha=0.3,
        edgecolors="none",
    )
    ax.set_zlim(z_min, z_max)
    ax.view_init(**LANDSCAPE_CAMERA)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _row_z_limits(
    meshes: dict[tuple[str, str], dict], task: str, models: list[tuple[str, str]]
) -> tuple[float, float]:
    stack = np.concatenate(
        [meshes[(task, label)]["zz"].ravel() for _, label in models]
    )
    z_min, z_max = float(stack.min()), float(stack.max())
    if meshes[(task, models[0][1])]["task_type"] == "classification":
        z_min, z_max = min(z_min, 0.0), max(z_max, 1.0)
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0
    return z_min, z_max


def make_decision_landscape_grid_3d(
    output_dir: Path,
    meshes: dict[tuple[str, str], dict],
    *,
    tasks: list[str],
    models: list[tuple[str, str]],
) -> Path:
    output_path = output_dir / "decision_landscape_grid_3d.png"
    fig, axes = plt.subplots(
        len(tasks),
        len(models),
        figsize=(len(models) * 2.6, len(tasks) * 2.5),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )

    for row, task in enumerate(tasks):
        z_min, z_max = _row_z_limits(meshes, task, models)
        for col, (_, label) in enumerate(models):
            ax = axes[row][col]
            _draw_landscape_panel(ax, meshes[(task, label)], z_min=z_min, z_max=z_max)
            if row == 0:
                ax.set_title(label, fontsize=10, pad=0)
            if col == 0:
                ax.text2D(
                    -0.12,
                    0.5,
                    task.replace("synthetic_", ""),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    fig.suptitle(
        "Decision Landscapes On True 2D Synthetic Tasks (height = predicted score)",
        fontsize=14,
        y=0.995,
    )
    top = 0.90 if len(tasks) > 1 else 0.78
    fig.subplots_adjust(left=0.03, right=0.99, top=top, bottom=0.02, wspace=0.02, hspace=0.05)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_landscape_orbit_gifs(
    output_dir: Path,
    meshes: dict[tuple[str, str], dict],
    *,
    tasks: list[str],
    models: list[tuple[str, str]],
) -> list[Path]:
    try:
        import PIL  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Orbit GIF export requires Pillow. Install with: pip install -e '.[viz]'"
        ) from exc

    paths: list[Path] = []
    azimuths = np.linspace(-180.0, 165.0, 24)
    for task in tasks:
        output_path = output_dir / f"decision_landscape_{task}_orbit.gif"
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(len(models) * 2.0, 2.4),
            squeeze=False,
            subplot_kw={"projection": "3d"},
        )
        z_min, z_max = _row_z_limits(meshes, task, models)
        for col, (_, label) in enumerate(models):
            ax = axes[0][col]
            _draw_landscape_panel(ax, meshes[(task, label)], z_min=z_min, z_max=z_max)
            ax.set_title(label, fontsize=7, pad=0)
        fig.suptitle(task.replace("synthetic_", ""), fontsize=10)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.02)

        def update(azim: float, axes_row=axes[0]):
            for ax in axes_row:
                ax.view_init(elev=LANDSCAPE_CAMERA["elev"], azim=azim)
            return []

        anim = animation.FuncAnimation(fig, update, frames=azimuths, blit=False)
        anim.save(output_path, writer=animation.PillowWriter(fps=8), dpi=90)
        plt.close(fig)
        paths.append(output_path)
    return paths


def export_landscape_html(
    output_dir: Path,
    meshes: dict[tuple[str, str], dict],
    *,
    tasks: list[str],
    models: list[tuple[str, str]],
) -> list[Path]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Interactive HTML export requires plotly. Install with: pip install -e '.[viz3d]'"
        ) from exc

    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for task in tasks:
        fig = make_subplots(
            rows=1,
            cols=len(models),
            specs=[[{"type": "surface"}] * len(models)],
            subplot_titles=[label for _, label in models],
        )
        z_min, z_max = _row_z_limits(meshes, task, models)
        for col, (_, label) in enumerate(models, start=1):
            entry = meshes[(task, label)]
            fig.add_trace(
                go.Surface(
                    x=entry["xx"],
                    y=entry["yy"],
                    z=entry["zz"],
                    colorscale="Viridis",
                    cmin=z_min,
                    cmax=z_max,
                    showscale=col == len(models),
                ),
                row=1,
                col=col,
            )
        fig.update_layout(
            title=f"Decision landscapes: {task} (research artifact, not a trading claim)",
            height=420,
            width=320 * len(models),
        )
        output_path = html_dir / f"decision_landscape_{task}.html"
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        paths.append(output_path)
    return paths


def make_gate_bar_field_3d(
    output_dir: Path,
    *,
    seeds: list[int],
    n_samples: int,
    max_epochs: int,
    n_features: int,
) -> Path:
    """Intended vs learned per-feature gate sparsity on the junk-feature task."""
    output_path = output_dir / "gate_bar_field_3d.png"
    dataset_name = "synthetic_irrelevant_noise"

    intended = np.zeros(n_features, dtype="float32")
    for idx in IRRELEVANT_NOISE_SIGNAL_FEATURES:
        if idx < n_features:
            intended[idx] = 1.0
    rows: list[tuple[str, np.ndarray]] = [("intended sparsity", intended)]

    for label in GATE_MODELS:
        for seed in seeds:
            bundle = load_dataset(
                dataset_name, split_seed=seed, n_samples=n_samples, n_features=n_features
            )
            config = _make_tst_config(label, bundle.task_type, seed, max_epochs)
            trained = Trainer(config, max_epochs=max_epochs, batch_size=128).fit(bundle)
            X_val = trained.preprocessor.transform(bundle.X_val).astype("float32")
            gate_values = extract_gate_values(trained.model, X_val)
            rows.append((f"{label} seed={seed}", _feature_vector(gate_values, n_features)))

    fig = plt.figure(figsize=(13.0, 7.5))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("viridis")
    xs = np.arange(n_features)
    for row_idx, (name, values) in enumerate(rows):
        heights = np.clip(values, 0.0, 1.0)
        finite = np.isfinite(heights)
        active = finite & (heights > 0.02)
        missing = ~finite
        if active.any():
            if name == "intended sparsity":
                colors = ["#c0392b"] * int(active.sum())
            else:
                colors = [cmap(h) for h in heights[active]]
            ax.bar3d(
                xs[active],
                np.full(int(active.sum()), row_idx, dtype="float32"),
                np.zeros(int(active.sum())),
                0.7,
                0.5,
                heights[active],
                color=colors,
                shade=True,
            )
        if missing.any():
            ax.scatter(
                xs[missing],
                np.full(int(missing.sum()), row_idx + 0.25, dtype="float32"),
                np.zeros(int(missing.sum())),
                c="#7f8c8d",
                marker="x",
                s=18,
                linewidths=1.0,
                depthshade=False,
            )
    ax.set_yticks(np.arange(len(rows)) + 0.25)
    ax.set_yticklabels([name for name, _ in rows], fontsize=7)
    ax.set_xticks(xs[:: max(1, n_features // 10)])
    ax.set_xticklabels(
        [f"x{i}" for i in xs[:: max(1, n_features // 10)]], fontsize=7
    )
    ax.set_zlim(0.0, 1.0)
    ax.set_zlabel("gate value")
    ax.view_init(elev=24, azim=-55)
    ax.set_title(
        "Intended vs Learned Feature-Gate Sparsity (synthetic_irrelevant_noise)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


BIAS_ATLAS_STAGES = [
    ("Raw tabular row  (B x F)", "core"),
    ("Feature tokens + column identity  (B x F x d)", "core"),
    ("Sparse feature gate  (opt-in: v1+)", "optional"),
    ("Fourier expansion  (opt-in: v2+)", "optional"),
    ("Interaction block: attention over feature tokens", "core"),
    ("Pooled state  (B x d)", "core"),
    ("Head: simple or regime MoE  (opt-in: v3)", "optional"),
]

BIAS_ATLAS_LADDER = {
    "TST-v0": [0, 0, 0],
    "TST-v1": [1, 0, 0],
    "TST-v2": [1, 1, 0],
    "TST-v3": [1, 1, 1],
}


def _draw_bias_signature_panels(fig, gs) -> None:
    xx, yy = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))

    ax_tree = fig.add_subplot(gs[1, 1], projection="3d")
    tree_z = ((xx > 0.6) & (yy < 0.4)).astype(float) * 0.9 + (
        (xx < 0.3) & (yy > 0.7)
    ).astype(float) * 0.45
    ax_tree.plot_surface(xx, yy, tree_z, cmap="viridis", rstride=1, cstride=1, linewidth=0)
    ax_tree.set_title("Tree bias:\naxis-aligned partitions", fontsize=9)

    ax_mlp = fig.add_subplot(gs[1, 2], projection="3d")
    mlp_z = 1.0 / (1.0 + np.exp(-8.0 * (xx + yy - 1.0)))
    ax_mlp.plot_surface(xx, yy, mlp_z, cmap="viridis", rstride=1, cstride=1, linewidth=0)
    ax_mlp.set_title("MLP bias:\nsmooth rotated mixing", fontsize=9)

    ax_tst = fig.add_subplot(gs[1, 3], projection="3d")
    gate_heights = np.array([0.95, 0.9, 0.85, 0.12, 0.1, 0.08, 0.1, 0.09, 0.11, 0.1])
    cmap = plt.get_cmap("viridis")
    ax_tst.bar3d(
        np.arange(len(gate_heights)),
        np.zeros(len(gate_heights)),
        np.zeros(len(gate_heights)),
        0.7,
        0.7,
        gate_heights,
        color=[cmap(h) for h in gate_heights],
        shade=True,
    )
    ax_tst.set_ylim(0, 4)
    ax_tst.set_zlim(0, 1)
    ax_tst.set_title("TST bias:\nper-feature tokens + gates", fontsize=9)

    for ax in (ax_tree, ax_mlp, ax_tst):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=26, azim=-60)


def make_bias_atlas(output_dir: Path) -> Path:
    """Layered tensor-topology diagram plus the TST ablation module ladder."""
    output_path = output_dir / "bias_atlas.png"
    fig = plt.figure(figsize=(15.0, 10.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1.0], hspace=0.18, wspace=0.28)

    ax_flow = fig.add_subplot(gs[0, :], projection="3d")
    edge_colors = {"core": "#53687e", "optional": "#c2703d"}
    face_colors = {"core": "#dce6f2", "optional": "#f9e4d4"}
    for i, (label, kind) in enumerate(BIAS_ATLAS_STAGES):
        z = float(i)
        verts = [[(0, 0, z), (4, 0, z), (4, 2, z), (0, 2, z)]]
        ax_flow.add_collection3d(
            Poly3DCollection(
                verts,
                facecolors=face_colors[kind],
                edgecolors=edge_colors[kind],
                alpha=0.85,
                linewidths=1.2,
            )
        )
        ax_flow.text(5.4, 2.4, z, label, fontsize=9, va="center", color="#2c3e50", zorder=20)
        if i:
            ax_flow.plot([2.0, 2.0], [1.0, 1.0], [z - 1.0, z], color="#34495e", lw=1.2)
    for t in range(6):
        x0 = 0.35 + 0.6 * t
        verts = [[(x0, 0.4, 1.03), (x0 + 0.45, 0.4, 1.03), (x0 + 0.45, 1.6, 1.03), (x0, 1.6, 1.03)]]
        ax_flow.add_collection3d(
            Poly3DCollection(verts, facecolors="#8fb3d9", edgecolors="#53687e", linewidths=0.8)
        )
    ax_flow.set_xlim(0, 8.5)
    ax_flow.set_ylim(0, 2.5)
    ax_flow.set_zlim(0, len(BIAS_ATLAS_STAGES))
    ax_flow.view_init(elev=16, azim=-72)
    ax_flow.set_axis_off()
    ax_flow.set_title(
        "Tabular State Transformer: Layered Tensor Topology (research artifact)",
        fontsize=14,
    )

    ax_ladder = fig.add_subplot(gs[1, 0])
    modules = ["Sparse gate", "Fourier", "MoE head"]
    lit = np.array(list(BIAS_ATLAS_LADDER.values()), dtype=float)
    ax_ladder.imshow(lit, cmap="Blues", vmin=0.0, vmax=1.5, aspect="auto")
    ax_ladder.set_xticks(np.arange(len(modules)))
    ax_ladder.set_xticklabels(modules, fontsize=8)
    ax_ladder.set_yticks(np.arange(len(BIAS_ATLAS_LADDER)))
    ax_ladder.set_yticklabels(list(BIAS_ATLAS_LADDER), fontsize=9)
    for r, row_values in enumerate(lit):
        for c, value in enumerate(row_values):
            ax_ladder.text(
                c,
                r,
                "on" if value else "-",
                ha="center",
                va="center",
                fontsize=9,
                color="#1a3552" if value else "#95a5a6",
            )
    ax_ladder.set_title("Ablation module ladder", fontsize=10)

    _draw_bias_signature_panels(fig, gs)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_tensor_topology_doc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Tensor Topology

These figures make the Tabular State Transformer representation legible. They are research
artifacts, not claims of model superiority.

## 2D figures

- `reports/figures/tensor_topology_schematic.png`: raw row to feature tokens, gates, spectral expansion, interaction block, pooled state, and head.
- `reports/figures/representation_shape_diagram.png`: tensor shape flow from `Batch x Features` to `Batch x d_model`.
- `reports/figures/feature_gate_heatmap.png`: learned sparse-gate values for TST-v1/TST-v2 on synthetic stress tasks.
- `reports/figures/synthetic_task_topology_grid.png`: decision surfaces for true 2D synthetic tasks only.

## 3D inductive-bias figures

- `reports/figures/3d/decision_landscape_grid_3d.png`: hero grid. Rows are true-2D synthetic
  tasks, columns are model families (Linear, RandomForest, LightGBM, MLP, FT-Transformer,
  TST-v0..v3), height is the predicted score / class probability. Each family's inductive bias
  should be readable from the surface shape alone: flat tilted planes (linear), axis-aligned
  plateaus and sharp ridges (trees), smooth rotated ramps (MLP), and token-attention surfaces
  (FT / TST ablations).
- `reports/figures/3d/decision_landscape_<task>_orbit.gif`: short orbit render per task row,
  same camera grammar as the static grid.
- `reports/figures/3d/gate_bar_field_3d.png`: intended vs learned per-feature gate sparsity on
  `synthetic_irrelevant_noise` (signal features are `x0, x1, x2`).
- `reports/figures/3d/bias_atlas.png`: layered tensor-topology diagram plus the TST-v0..v3
  module ladder and analytic bias-signature panels (tree partition vs MLP mixing vs TST tokens).
- `reports/figures/3d/meshes/<task>__<model>.npz`: exported meshes (`xx`, `yy`, `zz`,
  `x_train`, `y_train`, `task_type`) so downstream work can re-render without re-fitting.
- `reports/figures/3d/html/decision_landscape_<task>.html`: interactive plotly surfaces,
  generated with `--export-html` (requires the `viz3d` extra).

## Generation

```bash
# Existing 2D figures
venv/bin/python scripts/visualize_topology.py \\
  --mode 2d \\
  --results-csv reports/experiments/legacy/results.csv \\
  --output-dir reports/figures \\
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \\
  --seeds 42,43,44 \\
  --n-samples 1024 \\
  --max-epochs 20

# 3D decision landscapes + orbit GIFs + gate bar field (add --export-html for plotly HTML)
venv/bin/python scripts/visualize_topology.py \\
  --mode 3d-surfaces \\
  --output-dir reports/figures \\
  --tasks synthetic_xor,synthetic_piecewise,synthetic_axis_threshold,synthetic_rotated \\
  --seeds 42,43,44 \\
  --n-samples 1024 \\
  --max-epochs 20

# Bias atlas (no model fitting required)
venv/bin/python scripts/visualize_topology.py --mode bias-atlas --output-dir reports/figures

# Everything
venv/bin/python scripts/visualize_topology.py --mode all --output-dir reports/figures
```

Gate values are extracted through `extract_gate_values(model, X_valid)`: global gates use
`sigmoid(logits)`, input-dependent gates can expose or cache validation-set activations, and missing
gates are represented as `NaN`.

## Honesty rules

- Decision surfaces and landscapes are intentionally limited to native two-feature tasks.
  Higher-dimensional tasks are not projected into 2D or 3D here, because that would make the
  visualization look more faithful than it is. High-dimensional stress tasks appear only as
  metric / diagnostic panels (gate bar field, heatmaps, benchmark tables).
- Captions must describe these as research artifacts from an ablation harness, never as
  evidence of model superiority or trading claims (see `docs/finance_disclaimer.md`).

## Future visual continuity

- Track A bridge: financial-state tensor and graph figures made in the private Track A repo
  should reuse the same camera grammar (`elev=28, azim=-60`), the viridis surface scale, and
  the "bias signature" caption style so cross-track figures read as one research program.
- Neural ODE / SDE fork: continuous-time models live in a separate future fork, not this repo.
  Candidate panels there are trajectory flows and stochastic path ensembles rendered over the
  same synthetic landscapes; the `.npz` mesh exports exist so that fork can re-render Track B
  baselines without re-fitting them.
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
- ![3D decision landscape grid](figures/3d/decision_landscape_grid_3d.png)

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
        "--mode",
        default="2d",
        choices=["2d", "3d-surfaces", "bias-atlas", "all"],
        help="Which figure family to generate.",
    )
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
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="Also export interactive plotly HTML landscapes (requires the viz3d extra).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_ints(args.seeds)
    tasks = _parse_tasks(args.tasks)

    paths: list[Path] = []
    results_rows = 0

    if args.mode in {"2d", "all"}:
        result_rows = _load_result_rows(Path(args.results_csv))
        results_rows = len(result_rows)
        paths += [
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

    output_dir_3d = output_dir / "3d"

    if args.mode in {"3d-surfaces", "all"}:
        output_dir_3d.mkdir(parents=True, exist_ok=True)
        models = _resolve_3d_models()
        meshes = _compute_landscape_meshes(
            output_dir_3d,
            tasks=tasks,
            models=models,
            seed=args.surface_seed,
            n_samples=args.n_samples,
            max_epochs=args.max_epochs,
            grid_size=args.grid_size,
        )
        paths.append(
            make_decision_landscape_grid_3d(output_dir_3d, meshes, tasks=tasks, models=models)
        )
        paths.append(
            make_gate_bar_field_3d(
                output_dir_3d,
                seeds=seeds,
                n_samples=args.n_samples,
                max_epochs=args.max_epochs,
                n_features=args.gate_n_features,
            )
        )
        paths += make_landscape_orbit_gifs(output_dir_3d, meshes, tasks=tasks, models=models)
        if args.export_html:
            paths += export_landscape_html(output_dir_3d, meshes, tasks=tasks, models=models)

    if args.mode in {"bias-atlas", "all"}:
        output_dir_3d.mkdir(parents=True, exist_ok=True)
        paths.append(make_bias_atlas(output_dir_3d))

    _write_tensor_topology_doc(Path(args.docs_output))
    if args.mode in {"2d", "all"}:
        _update_benchmark_links(Path(args.benchmark_report))
    print(
        {
            "mode": args.mode,
            "figures": [str(path) for path in paths],
            "docs": args.docs_output,
            "results_rows": results_rows,
            "surface_tasks": tasks,
        }
    )


if __name__ == "__main__":
    main()
