#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from spectraloom_experiments.io import write_csv, write_json
from spectraloom_experiments.protocols import TASK_FILES, iter_records, load_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ZuCo corpus structure and truncation/leakage risks.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_FILES), required=True)
    parser.add_argument(
        "--pickle-dir", type=Path, default=REPO / "EEG-To-text" / "Data" / "pickle_file"
    )
    parser.add_argument("--max-length", type=int, default=56)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dataset_audit")
    return parser.parse_args()


def summarize_task(task: str, dataset: dict, max_length: int) -> tuple[dict, list[dict]]:
    rows = list(iter_records(dataset, task))
    reference_counts = Counter(row["reference"] for row in rows)
    subject_counts = Counter(row["subject"] for row in rows)
    subject_sentence_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject_sentence_sets[row["subject"]].add(row["reference"])
    common = set.intersection(*subject_sentence_sets.values()) if subject_sentence_sets else set()
    lengths = [row["word_count"] for row in rows]
    summary = {
        "task": task,
        "subjects": len(subject_counts),
        "recordings": len(rows),
        "unique_sentences": len(reference_counts),
        "sentences_read_by_all_subjects": len(common),
        "mean_readings_per_sentence": len(rows) / max(len(reference_counts), 1),
        "max_readings_per_sentence": max(reference_counts.values(), default=0),
        "min_subject_recordings": min(subject_counts.values(), default=0),
        "max_subject_recordings": max(subject_counts.values(), default=0),
        "mean_fixated_word_count": sum(lengths) / max(len(lengths), 1),
        "max_fixated_word_count": max(lengths, default=0),
        "recordings_over_max_input_length": sum(n > max_length for n in lengths),
        "recordings_at_or_over_max_input_length": sum(n >= max_length for n in lengths),
        "empty_word_sequences": sum(n == 0 for n in lengths),
    }
    compact_rows = [
        {key: value for key, value in row.items() if key != "sent_obj"}
        for row in rows
    ]
    return summary, compact_rows


def main() -> None:
    args = parse_args()
    summaries = []
    all_rows = []
    sentence_tasks: dict[str, set[str]] = defaultdict(set)
    for task in args.tasks:
        dataset = load_task(args.pickle_dir, task)
        summary, rows = summarize_task(task, dataset, args.max_length)
        summaries.append(summary)
        all_rows.extend(rows)
        for row in rows:
            sentence_tasks[row["reference"]].add(task)
    cross_task = [
        {"reference": reference, "tasks": "|".join(sorted(tasks)), "task_count": len(tasks)}
        for reference, tasks in sentence_tasks.items()
        if len(tasks) > 1
    ]
    payload = {
        "tasks": summaries,
        "cross_task_duplicate_sentence_count": len(cross_task),
        "warning": (
            "word_count is the number of retained fixation-bearing words, so the input mask is a "
            "potential linguistic side channel and is not proof of neural information."
        ),
    }
    write_json(args.output_dir / "summary.json", payload)
    write_csv(args.output_dir / "task_summary.csv", summaries)
    write_csv(args.output_dir / "recordings.csv", all_rows)
    write_csv(args.output_dir / "cross_task_duplicates.csv", cross_task)
    print(payload)


if __name__ == "__main__":
    main()

