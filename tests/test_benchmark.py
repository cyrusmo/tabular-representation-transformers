from __future__ import annotations

from tabular_state_transformer.evaluation import run_benchmark
from tabular_state_transformer.evaluation.benchmark import DEFAULT_BASELINES


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
