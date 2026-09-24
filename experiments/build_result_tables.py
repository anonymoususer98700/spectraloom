#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("wer", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l_f1", "corpus_bleu", "chrf")


def rows(summary: Path, phase: str) -> list[dict]:
    values = json.loads(summary.read_text(encoding="utf-8"))
    grouped: dict[str, dict] = {}
    for item in values:
        row = grouped.setdefault(item["variant"], {"phase": phase, "variant": item["variant"], "seeds": item["seeds"]})
        if item["metric"] in METRICS:
            row[item["metric"]] = item["mean"]
            row[item["metric"] + "_sd"] = item["sd"]
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    table_dir = args.outputs / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    all_rows = rows(args.outputs / "in_domain_summary.json", "zuco1_test")
    all_rows += rows(args.outputs / "zero_shot_summary.json", "zuco2_zero_shot")
    fields = ["phase", "variant", "seeds"] + [name for metric in METRICS for name in (metric, metric + "_sd")]
    with (table_dir / "all_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (table_dir / "all_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(table_dir / "all_results.csv")


if __name__ == "__main__":
    main()
