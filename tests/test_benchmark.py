from __future__ import annotations

from tabular_state_transformer.evaluation import run_benchmark


def test_benchmark_script_smoke(tmp_path):
    output = tmp_path / "benchmark.md"
    results = run_benchmark("synthetic", output_path=output, n_samples=64, max_epochs=1)
    assert results
    assert output.exists()
