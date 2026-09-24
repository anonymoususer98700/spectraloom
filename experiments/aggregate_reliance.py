"""Aggregate matched EEG-reliance conditions across training seeds."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from spectraloom_experiments.reliance import CONDITIONS


METRICS = ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l_f1", "wer", "cer", "chrf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 312])
    parser.add_argument("--variant", default="spectraloom")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Repeated seeds")

    rows = []
    for phase in ("test", "zero_shot"):
        runs = {}
        for seed in args.seeds:
            folder = args.training_dir / f"{args.variant}_seed{seed}" / "reliance" / phase
            runs[seed] = {}
            reference_ids = None
            for condition in CONDITIONS:
                with (folder / f"{condition}.csv").open(encoding="utf-8-sig", newline="") as handle:
                    ids = [row["example_id"] for row in csv.DictReader(handle)]
                if reference_ids is not None and ids != reference_ids:
                    raise ValueError(f"Unpaired examples: {folder} / {condition}")
                reference_ids = ids
                runs[seed][condition] = json.loads((folder / f"{condition}.metrics.json").read_text())
        for condition in CONDITIONS:
            for metric in METRICS:
                values = [runs[seed][condition][metric] for seed in args.seeds]
                sign = -1 if metric in ("wer", "cer") else 1
                advantages = [
                    sign * (runs[seed]["real"][metric] - runs[seed][condition][metric])
                    for seed in args.seeds
                ]
                rows.append({
                    "phase": phase, "condition": condition, "metric": metric,
                    "seeds": args.seeds, "seed_values": values,
                    "mean": statistics.mean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else None,
                    "paired_real_advantage": advantages,
                    "mean_real_advantage": statistics.mean(advantages),
                    "sd_real_advantage": statistics.stdev(advantages) if len(advantages) > 1 else None,
                    "eligible_examples": [runs[seed][condition]["eligible_examples"] for seed in args.seeds],
                })
    output = args.output or args.training_dir.parent / "reliance_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
