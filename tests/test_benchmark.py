from __future__ import annotations

from tabular_state_transformer.evaluation import run_benchmark
from tabular_state_transformer.evaluation.benchmark import (
    DEFAULT_BASELINES,
    TuningCandidate,
    _select_best_tuning_candidate,
    _selected_config_id,
)


def test_benchmark_script_smoke(tmp_path):
    output = tmp_path / "benchmark.md"
    diagnostics_output = tmp_path / "diagnostics.csv"
    results = run_benchmark(
        "synthetic",
        output_path=output,
        diagnostics_output_path=diagnostics_output,
        n_samples=64,
        max_epochs=1,
        dataset_names=["synthetic_xor"],
        baselines=["linear"],
        model_configs=["TST-v0"],
    )
    assert results
    assert output.exists()
    assert results[0].seed == 42
    row = results[0].as_row()
    assert row["status"] == "ok"
    assert "error_message" in row
    assert "predict_seconds" in row
    assert row["n_samples"] == 64
    assert row["n_features"] == 20
    assert diagnostics_output.exists()
    diagnostic_text = diagnostics_output.read_text()
    assert "effective_training_status" in diagnostic_text
    assert "best_epoch" in diagnostic_text


def test_benchmark_writes_csv_and_handles_multiple_seeds(tmp_path):
    output = tmp_path / "benchmark.md"
    csv_output = tmp_path / "benchmark.csv"
    results = run_benchmark(
        "synthetic_stress",
        output_path=output,
        csv_output_path=csv_output,
        n_samples=64,
        max_epochs=1,
        seeds=[42, 43],
        dataset_names=["synthetic_xor"],
        baselines=["linear"],
        model_configs=[],
    )
    assert [result.seed for result in results] == [42, 43]
    assert output.exists()
    assert csv_output.exists()
    text = csv_output.read_text()
    assert "status" in text
    assert "error_message" in text


def test_default_baselines_include_safe_gradient_boosting_not_xgboost():
    assert "gradient_boosting" in DEFAULT_BASELINES
    assert "xgboost" not in DEFAULT_BASELINES


def test_ft_transformer_style_neural_baseline_is_opt_in(tmp_path):
    output = tmp_path / "benchmark.md"
    diagnostics_output = tmp_path / "diagnostics.csv"
    results = run_benchmark(
        "synthetic",
        output_path=output,
        diagnostics_output_path=diagnostics_output,
        n_samples=64,
        max_epochs=1,
        dataset_names=["synthetic_xor"],
        baselines=[],
        model_configs=[],
        neural_baselines=["ft_transformer"],
    )

    assert len(results) == 1
    row = results[0].as_row()
    assert row["model"] == "FT-Transformer-style"
    assert row["variant"] == "local_ft_transformer"
    assert row["family"] == "neural_baseline"
    assert row["benchmark_mode"] == "default_benchmark"
    assert diagnostics_output.exists()
    assert "effective_training_status" in diagnostics_output.read_text()


def test_tuned_tst_benchmark_emits_audit_metadata(tmp_path):
    output = tmp_path / "benchmark.md"
    diagnostics_output = tmp_path / "diagnostics.csv"
    results = run_benchmark(
        "synthetic",
        output_path=output,
        diagnostics_output_path=diagnostics_output,
        n_samples=64,
        max_epochs=1,
        tuning_max_epochs=1,
        benchmark_mode="tuned_tst_benchmark",
        dataset_names=["synthetic_xor"],
        baselines=[],
        model_configs=["TST-v0"],
    )

    rows = [result.as_row() for result in results]
    tuned = next(row for row in rows if row["variant"] == "TST-v0-tuned")
    assert tuned["model"] == "TST"
    assert tuned["base_variant"] == "TST-v0"
    assert tuned["selected_config_id"].startswith("tst_v0_lr_")
    assert tuned["selection_metric"] == "val_accuracy"
    assert tuned["selection_mode"] == "maximize"
    assert tuned["candidate_config_count"] == "3"
    assert tuned["candidate_lrs"] == "1e-4,3e-4,1e-3"
    assert tuned["tuning_budget_type"] == "lr_only"
    assert diagnostics_output.exists()
    assert "TST-v0-tuned" in diagnostics_output.read_text()


def test_tuned_selection_ids_and_tie_breaking_are_deterministic():
    assert _selected_config_id("TST-v1-Gate", 3e-4) == "tst_v1_gate_lr_3e-4"
    assert _selected_config_id("TST-v2-GateFourier", 3e-4) == "tst_v2_gate_fourier_lr_3e-4"
    assert _selected_config_id("TST-v3-MoE", 1e-4) == "tst_v3_moe_lr_1e-4"
    candidates = [
        TuningCandidate(object(), 1e-3, 2, "late_high_lr", 1.0, "val_accuracy", "maximize", 0.8, 2),
        TuningCandidate(object(), 3e-4, 1, "middle_lr", 1.0, "val_accuracy", "maximize", 0.8, 2),
        TuningCandidate(object(), 1e-4, 0, "low_lr", 1.0, "val_accuracy", "maximize", 0.8, 3),
    ]
    selected = _select_best_tuning_candidate("classification", candidates)
    assert selected.config_id == "middle_lr"
