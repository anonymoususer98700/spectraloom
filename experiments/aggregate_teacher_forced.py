#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate teacher-forced metrics by model.")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in args.training_dir.glob("*/teacher_forced_test_predictions.metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = str(payload["run_name"])
        model = run_name.rsplit("_seed", 1)[0]
        grouped[model].append(payload)
    if not grouped:
        raise SystemExit("No teacher-forced test metrics found")
    fields = (
        "bleu_1", "bleu_2", "bleu_3", "bleu_4", "chrf", "rouge_l_f1", "wer",
        "teacher_forced_token_accuracy", "teacher_forced_nll", "teacher_forced_perplexity",
    )
    result = []
    for model, rows in sorted(grouped.items()):
        item: dict = {"model": model, "seeds": sorted(int(row["seed"]) for row in rows), "n_seeds": len(rows)}
        for field in fields:
            values = [float(row[field]) for row in rows]
            item[field] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        result.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
