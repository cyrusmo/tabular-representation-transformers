from __future__ import annotations

from tabular_state_transformer.evaluation import run_benchmark


def test_benchmark_script_smoke(tmp_path):
    output = tmp_path / "benchmark.md"
    results = run_benchmark(
        "synthetic",
        output_path=output,
        n_samples=64,
        max_epochs=1,
        dataset_names=["synthetic_xor"],
        baselines=["linear"],
        model_configs=["TST-v0"],
    )
    assert results
    assert output.exists()
    assert results[0].seed == 42


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
