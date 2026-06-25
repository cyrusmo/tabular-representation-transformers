from __future__ import annotations

import argparse

from tabular_state_transformer.evaluation.trainability import run_trainability_audit


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_int_csv(value: str | None) -> list[int] | None:
    items = _split_csv(value)
    if items is None:
        return None
    return [int(item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/trainability_audit_results.md")
    parser.add_argument("--output-csv", default="reports/trainability_audit_results.csv")
    parser.add_argument("--diagnostics-output", default="reports/trainability_audit_diagnostics.csv")
    parser.add_argument("--tasks", default=None, help="Comma-separated audit tasks")
    parser.add_argument("--models", default=None, help="Comma-separated models")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    results = run_trainability_audit(
        output_path=args.output,
        csv_output_path=args.output_csv,
        diagnostics_output_path=args.diagnostics_output,
        task_names=_split_csv(args.tasks),
        model_names=_split_csv(args.models),
        seeds=_split_int_csv(args.seeds),
        n_samples=args.n_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        continue_on_error=not args.fail_fast,
    )
    print({"results": len(results), "output": args.output})


if __name__ == "__main__":
    main()
