#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from spectraloom_experiments.io import sentence_id, write_csv, write_json
from spectraloom_experiments.protocols import TASK_FILES, iter_records, load_task, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create source-train/target-test cross-corpus protocol.")
    parser.add_argument("--source-tasks", nargs="+", choices=sorted(TASK_FILES), required=True)
    parser.add_argument("--target-tasks", nargs="+", choices=sorted(TASK_FILES), required=True)
    parser.add_argument("--pickle-dir", type=Path, default=REPO / "EEG-To-text" / "Data" / "pickle_file")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if set(args.source_tasks) & set(args.target_tasks):
        raise ValueError("Source and target tasks must be disjoint")
    source = [
        row for task in args.source_tasks for row in iter_records(load_task(args.pickle_dir, task), task)
    ]
    target = [
        row for task in args.target_tasks for row in iter_records(load_task(args.pickle_dir, task), task)
    ]
    target_texts = {row["reference"] for row in target}
    source_texts = sorted(
        {row["reference"] for row in source if row["reference"] not in target_texts},
        key=lambda text: hashlib.sha256(f"{args.seed}\0{text}".encode()).hexdigest(),
    )
    train_end = int(0.8 * len(source_texts))
    dev_end = train_end + int(0.1 * len(source_texts))
    train_texts = set(source_texts[:train_end])
    dev_texts = set(source_texts[train_end:dev_end])
    rows = []
    for row in source:
        if row["reference"] in target_texts:
            phase = "excluded"
        else:
            if row["reference"] in train_texts:
                phase = "train"
            elif row["reference"] in dev_texts:
                phase = "dev"
            else:
                phase = "test"
        rows.append(
            {k: v for k, v in row.items() if k != "sent_obj"}
            | {"phase": phase, "protocol": "cross_corpus", "split_seed": args.seed, "held_out_subject": ""}
        )
    for row in target:
        rows.append(
            {k: v for k, v in row.items() if k != "sent_obj"}
            | {"phase": "zero_shot", "protocol": "cross_corpus", "split_seed": args.seed, "held_out_subject": ""}
        )
    report = validate_manifest(rows)
    report["source_target_text_overlap_excluded"] = len({r["reference"] for r in source} & target_texts)
    write_csv(args.output, rows)
    write_json(args.output.with_suffix(".summary.json"), report)
    print(report)


if __name__ == "__main__":
    main()
