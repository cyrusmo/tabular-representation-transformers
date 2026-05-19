from __future__ import annotations

import csv
from pathlib import Path


def _columns(rows: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _header(key: str) -> str:
    return key.replace("_", " ").title()


def write_markdown_table(rows: list[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(rows) or ["model", "dataset", "metric", "score", "notes"]
    headers = [_header(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    output.write_text("\n".join(lines) + "\n")


def write_csv_table(rows: list[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(rows) or ["model", "dataset", "metric", "score", "notes"]
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
