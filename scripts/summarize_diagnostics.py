from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> str:
    return f"{mean(values):.6f}" if values else ""


def _read_final_rows(path: Path) -> list[dict[str, str]]:
    latest: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["seed"], row["family"], row["variant"])
            previous = latest.get(key)
            if previous is None or int(row.get("epoch") or 0) >= int(previous.get("epoch") or 0):
                latest[key] = row
    return list(latest.values())


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"], row["family"], row["variant"])].append(row)

    summary: list[dict[str, object]] = []
    for (dataset, task, family, variant), group in groups.items():
        val_metrics = [_float(row.get("final_val_metric", "")) for row in group]
        best_epochs = [_float(row.get("best_epoch", "")) for row in group]
        train_val_gaps = [_float(row.get("train_val_gap", "")) for row in group]
        gate_sparsities = [_float(row.get("gate_sparsity", "")) for row in group]
        statuses = Counter(row.get("effective_training_status", "") for row in group)
        summary.append(
            {
                "dataset": dataset,
                "task": task,
                "family": family,
                "variant": variant,
                "runs": len(group),
                "status_mode": statuses.most_common(1)[0][0] if statuses else "",
                "status_counts": ", ".join(
                    f"{status}:{count}" for status, count in sorted(statuses.items())
                ),
                "mean_final_val_metric": _mean([value for value in val_metrics if value is not None]),
                "mean_best_epoch": _mean([value for value in best_epochs if value is not None]),
                "mean_train_val_gap": _mean([value for value in train_val_gaps if value is not None]),
                "mean_gate_sparsity": _mean(
                    [value for value in gate_sparsities if value is not None]
                ),
            }
        )
    summary.sort(key=lambda row: (str(row["dataset"]), str(row["variant"])))
    return summary


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "task",
        "family",
        "variant",
        "runs",
        "status_mode",
        "status_counts",
        "mean_final_val_metric",
        "mean_best_epoch",
        "mean_train_val_gap",
        "mean_gate_sparsity",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, object]], path: Path, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(str(row["status_mode"]) for row in rows)
    lines = [
        "# TST Diagnostics Summary",
        "",
        f"Source: `{source}`. One row per final epoch for each dataset/seed/model variant run.",
        "",
        "## Status Counts",
        "",
        "| Status | Dataset/Variant Groups |",
        "| --- | --- |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Final-Epoch Aggregate",
            "",
            "| Dataset | Task | Family | Variant | Runs | Status Counts | Mean Final Val Metric | Mean Best Epoch | Mean Train-Val Gap | Mean Gate Sparsity |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {task} | {family} | {variant} | {runs} | {status_counts} | {mean_final_val_metric} | {mean_best_epoch} | {mean_train_val_gap} | {mean_gate_sparsity} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `early_stopped_cleanly` means validation checkpointing worked without numerical instability; it does not mean the model learned a useful boundary.",
            "- Flat metrics with best epoch near 1 are best interpreted as under-training or an inductive-bias mismatch, especially on rotated and irrelevant-noise tasks.",
            "- Gate sparsity is only populated for variants with a gate; blank cells mean the model has no gate rather than zero sparsity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/experiments/legacy/diagnostics.csv",
    )
    parser.add_argument(
        "--output-md",
        default="reports/analysis/diagnostics_summary.md",
    )
    parser.add_argument("--output-csv", default="reports/tables/diagnostics_summary.csv")
    args = parser.parse_args()

    final_rows = _read_final_rows(Path(args.input))
    rows = _aggregate(final_rows)
    _write_markdown(rows, Path(args.output_md), Path(args.input))
    _write_csv(rows, Path(args.output_csv))
    print({"rows": len(rows), "output_md": args.output_md, "output_csv": args.output_csv})


if __name__ == "__main__":
    main()
