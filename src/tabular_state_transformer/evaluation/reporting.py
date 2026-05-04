from __future__ import annotations

from pathlib import Path


def write_markdown_table(rows: list[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Dataset", "Metric", "Score", "Notes"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model", "")),
                    str(row.get("dataset", "")),
                    str(row.get("metric", "")),
                    str(row.get("score", "")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n")
