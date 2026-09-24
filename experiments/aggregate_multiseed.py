#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "bleu_1", "bleu_2", "bleu_3", "bleu_4", "corpus_bleu",
    "chrf", "rouge_l_f1", "token_f1", "wer", "cer",
)
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent training seeds, never recordings as seeds.")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("test", "zero_shot"), default="test")
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.training_dir.glob(f"*/{args.phase}_predictions.metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = SEED_RE.search(payload.get("run_name", path.parent.name))
        if not match:
            continue
        run_name = payload.get("run_name", path.parent.name)
        variant = re.sub(r"_seed\d+$", "", run_name, flags=re.IGNORECASE)
        groups[variant].append(payload)
    if not groups:
        raise SystemExit(f"No {args.phase}_predictions.metrics.json files found")
    rows = []
    for variant, runs in sorted(groups.items()):
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            # t critical values for 2--10 runs; normal approximation thereafter.
            critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(values), 1.96)
            half = critical * sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "seeds": len(values),
                    "mean": statistics.mean(values),
                    "sd": sd,
                    "ci95_low": statistics.mean(values) - half,
                    "ci95_high": statistics.mean(values) + half,
                    "seed_values": values,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
