from __future__ import annotations

from tabular_state_transformer.evaluation.trainability import run_trainability_audit


def test_trainability_audit_emits_schema_and_tst_diagnostics(tmp_path):
    output = tmp_path / "trainability.md"
    csv_output = tmp_path / "trainability.csv"
    diagnostics_output = tmp_path / "trainability_diagnostics.csv"

    rows = run_trainability_audit(
        output_path=output,
        csv_output_path=csv_output,
        diagnostics_output_path=diagnostics_output,
        task_names=["xor_2f"],
        model_names=["mlp", "TST-v0"],
        seeds=[42],
        n_samples=64,
        max_epochs=1,
    )

    assert len(rows) == 2
    assert output.exists()
    assert csv_output.exists()
    assert diagnostics_output.exists()
    assert "Trainability Verdict" in output.read_text()
    for row in rows:
        assert row["status"] == "ok"
        assert row["dataset"] == "synthetic_xor_2f"
        assert "train_accuracy" in row
        assert "val_accuracy" in row
        assert "test_accuracy" in row
        assert "train_loss" in row
        assert "logit_std" in row
        assert "prediction_class_balance" in row
        assert row["early_stopping"] == "disabled"
    diagnostic_text = diagnostics_output.read_text()
    assert "TST-v0" in diagnostic_text
    assert "train_metric" in diagnostic_text


def test_mlp_sanity_overfits_tiny_no_noise_xor(tmp_path):
    rows = run_trainability_audit(
        output_path=tmp_path / "trainability.md",
        csv_output_path=tmp_path / "trainability.csv",
        diagnostics_output_path=tmp_path / "trainability_diagnostics.csv",
        task_names=["xor_2f"],
        model_names=["mlp"],
        seeds=[42],
        n_samples=64,
        max_epochs=500,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "MLP"
    assert row["train_accuracy"] >= 0.95
