from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


REFERENCE_COLUMNS = ("reference", "target", "target_text", "gold", "text")
PREDICTION_COLUMNS = (
    "prediction",
    "predicted",
    "predicted_text",
    "predicted_text_no_tf",
    "hypothesis",
)
ID_COLUMNS = ("example_id", "sample_id", "id", "sample_index")


def canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def sentence_id(task: str, reference: str) -> str:
    # The same textual stimulus repeated across subjects or task files is one
    # inferential cluster; task-specific IDs would permit cross-task leakage.
    payload = canonical_text(reference).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def example_id(task: str, subject: str, reference: str, occurrence: int) -> str:
    payload = (
        f"{task}\0{subject}\0{canonical_text(reference)}\0{occurrence}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _choose(columns: Iterable[str], candidates: Iterable[str], explicit: str | None) -> str:
    cols = list(columns)
    if explicit:
        if explicit not in cols:
            raise ValueError(f"Column {explicit!r} not present; available: {cols}")
        return explicit
    for name in candidates:
        if name in cols:
            return name
    raise ValueError(f"None of {tuple(candidates)} found; available: {cols}")


def read_predictions(
    path: str | Path,
    reference_column: str | None = None,
    prediction_column: str | None = None,
    id_column: str | None = None,
    task: str = "unknown",
) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        ref_col = _choose(reader.fieldnames, REFERENCE_COLUMNS, reference_column)
        pred_col = _choose(reader.fieldnames, PREDICTION_COLUMNS, prediction_column)
        inferred_id = None
        if id_column:
            inferred_id = _choose(reader.fieldnames, ID_COLUMNS, id_column)
        else:
            inferred_id = next((c for c in ID_COLUMNS if c in reader.fieldnames), None)

        rows: list[dict[str, str]] = []
        seen: dict[tuple[str, str, str], int] = {}
        for index, row in enumerate(reader):
            reference = canonical_text(row.get(ref_col, ""))
            prediction = canonical_text(row.get(pred_col, ""))
            row_task = canonical_text(row.get("task", task)) or task
            subject = canonical_text(row.get("subject", "unknown")) or "unknown"
            key = (row_task, subject, reference)
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            eid = canonical_text(row.get(inferred_id, "")) if inferred_id else ""
            if not eid:
                eid = example_id(row_task, subject, reference, occurrence)
            sid = canonical_text(row.get("sentence_id", "")) or sentence_id(row_task, reference)
            rows.append(
                {
                    **{k: canonical_text(v) for k, v in row.items() if k},
                    "example_id": eid,
                    "sentence_id": sid,
                    "task": row_task,
                    "subject": subject,
                    "reference": reference,
                    "prediction": prediction,
                    "row_index": str(index),
                }
            )
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
