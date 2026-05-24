from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _score_sort_key(metric: str, score: float) -> float:
    return -score if metric == "accuracy" else score


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _read_success_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return [
            row
            for row in csv.DictReader(fh)
            if row.get("status") == "ok" and row.get("metric") and row.get("score")
        ]


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    wins: dict[tuple[str, str, str, str, str, str], int] = defaultdict(int)

    for row in rows:
        key = (
            row["dataset"],
            row["task"],
            row["metric"],
            row["family"],
            row["model"],
            row["variant"],
        )
        grouped[key].append(_float(row["score"]))

    per_seed_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        per_seed_groups[(row["dataset"], row["seed"], row["metric"])].append(row)

    for (_, _, metric), seed_rows in per_seed_groups.items():
        ordered = sorted(seed_rows, key=lambda row: _score_sort_key(metric, _float(row["score"])))
        if not ordered:
            continue
        best_score = _float(ordered[0]["score"])
        for row in ordered:
            score = _float(row["score"])
            if score != best_score:
                break
            key = (
                row["dataset"],
                row["task"],
                row["metric"],
                row["family"],
                row["model"],
                row["variant"],
            )
            wins[key] += 1

    summary_rows: list[dict[str, object]] = []
    for key, scores in grouped.items():
        dataset, task, metric, family, model, variant = key
        summary_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "metric": metric,
                "family": family,
                "model": model,
                "variant": variant,
                "mean_score": mean(scores),
                "std_score": _std(scores),
                "n_seeds": len(scores),
                "wins": wins[key],
            }
        )
    summary_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            _score_sort_key(str(row["metric"]), float(row["mean_score"])),
            str(row["model"]),
        )
    )
    return summary_rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "task",
        "metric",
        "family",
        "model",
        "variant",
        "mean_score",
        "std_score",
        "n_seeds",
        "wins",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "mean_score": _format_float(float(row["mean_score"])),
                    "std_score": _format_float(float(row["std_score"])),
                }
            )


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Seed Summary",
        "",
        "Scores are aggregated over successful benchmark rows. Accuracy is higher-better; RMSE is lower-better.",
        "",
        "| Dataset | Task | Metric | Model | Variant | Mean | Std | Seeds | Wins |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {task} | {metric} | {model} | {variant} | {mean} | {std} | {n_seeds} | {wins} |".format(
                dataset=row["dataset"],
                task=row["task"],
                metric=row["metric"],
                model=row["model"],
                variant=row["variant"],
                mean=_format_float(float(row["mean_score"])),
                std=_format_float(float(row["std_score"])),
                n_seeds=row["n_seeds"],
                wins=row["wins"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/benchmark_results.csv")
    parser.add_argument("--output-md", default="reports/tables/benchmark_seed_summary.md")
    parser.add_argument("--output-csv", default="reports/tables/benchmark_seed_summary.csv")
    args = parser.parse_args()

    rows = _aggregate(_read_success_rows(Path(args.input)))
    _write_markdown(rows, Path(args.output_md))
    _write_csv(rows, Path(args.output_csv))
    print({"rows": len(rows), "output_md": args.output_md, "output_csv": args.output_csv})


if __name__ == "__main__":
    main()
