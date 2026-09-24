#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from spectraloom_experiments.io import write_csv, write_json
from spectraloom_experiments.legacy_adapter import instantiate_legacy_model, load_checkpoint, seed_everything, tokenizer_name
from spectraloom_experiments.manifest_dataset import DEFAULT_BANDS, ManifestZuCoDataset
from spectraloom_experiments.metrics import compute_metrics
from spectraloom_experiments.reliance import CONDITIONS, paired_plan, perturb


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a corrected run on its immutable test manifest.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--phase", choices=("dev", "test", "zero_shot"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--min-new-tokens", type=int, default=0,
        help="Minimum generated continuation length; select on dev only.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--length-penalty", type=float, default=1.4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reliance-condition", choices=CONDITIONS)
    parser.add_argument("--perturbation-seed", type=int, default=2026)
    args = parser.parse_args()

    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    started_at = time.time()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    signature = None
    if args.reliance_condition:
        if config.get("train_control", "real") != "real":
            raise ValueError("Paired reliance requires a real-EEG-trained checkpoint")
        if config["normalization"] != "train_feature":
            raise ValueError("Gaussian reliance requires train-feature normalization")
        args.output = args.output or args.run_dir / "reliance" / args.phase / f"{args.reliance_condition}.csv"
        signature = {
            "checkpoint": file_hash(args.run_dir / args.checkpoint),
            "config": file_hash(args.run_dir / "config.json"),
            "manifest": file_hash(config["manifest"]),
            "normalization": file_hash(args.run_dir / "normalization.npz"),
            "evaluator": file_hash(Path(__file__)),
            "perturbations": file_hash(ROOT / "src" / "spectraloom_experiments" / "reliance.py"),
            "model_source": file_hash(REPO / "EEG-To-text" / "model_decoding.py"),
            "settings": {key: str(value) for key, value in vars(args).items()},
        }
        metrics_path = args.output.with_suffix(".metrics.json")
        if metrics_path.exists() and args.output.exists():
            try:
                previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                previous = {}
            if (previous.get("reliance_signature") == signature
                    and previous.get("predictions_sha256") == file_hash(args.output)):
                print(f"Skip completed reliance: {args.output}")
                return
    seed_everything(int(config["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_id", tokenizer_name(config["model"]))
    )
    mean = std = None
    stats_path = args.run_dir / "normalization.npz"
    if config["normalization"] == "train_feature":
        stats = np.load(stats_path)
        mean, std = stats["mean"], stats["std"]
    dataset = ManifestZuCoDataset(
        config["manifest"],
        args.phase,
        config["pickle_dir"],
        tokenizer,
        bands=list(DEFAULT_BANDS),
        max_length=int(config["max_length"]),
        normalization=config["normalization"],
        train_mean=mean,
        train_std=std,
    )
    model_config = {
        "model_name": config["model"],
        "eeg_bands": list(DEFAULT_BANDS),
        "ablate_sba": "uniform" if config.get("uniform_sba") else config.get("ablate_sba", False),
        "ablate_multiscale": config.get("ablate_multiscale", False),
        "ablate_cab": config.get("ablate_cab", False),
        "conv_kernels": config.get("conv_kernels", "3,5,7"),
        "dropout": 0.3,
    }
    model = instantiate_legacy_model(REPO, model_config, device)
    load_checkpoint(model, args.run_dir / args.checkpoint)
    model.eval()
    output_rows = []
    plan = paired_plan(dataset.inputs, args.perturbation_seed) if signature else None
    indices = [entry["index"] for entry in plan] if plan is not None else list(range(len(dataset)))
    with torch.inference_mode():
        starts = range(0, len(indices), args.batch_size)
        for start in tqdm(
            starts, total=len(starts), desc=f"Evaluate {args.phase}",
            unit="batch", dynamic_ncols=True,
        ):
            samples = [dataset.inputs[index] for index in indices[start : start + args.batch_size]]
            embeddings = torch.stack([s["input_embeddings"] for s in samples]).to(device).float()
            mask = torch.stack([s["input_attn_mask"] for s in samples]).to(device)
            inverse = torch.stack([s["input_attn_mask_invert"] for s in samples]).to(device)
            if plan is not None:
                changed = [
                    perturb(dataset.inputs, entry, args.reliance_condition, args.perturbation_seed)
                    for entry in plan[start : start + args.batch_size]
                ]
                embeddings, mask, inverse = [
                    torch.stack([item[column] for item in changed]).to(device)
                    for column in range(3)
                ]
            train_control = config.get("train_control", "real")
            if train_control == "zero_values_original_mask":
                embeddings.zero_()
            elif train_control == "zero_values_full_mask":
                embeddings.zero_(); mask.fill_(1); inverse.zero_()
            elif train_control == "gaussian_original_mask":
                embeddings = torch.randn_like(embeddings) * mask.unsqueeze(-1)
            targets = torch.stack([s["target_ids"] for s in samples]).to(device)
            labels = targets.masked_fill(targets == tokenizer.pad_token_id, -100)
            generated = model.generate(
                embeddings,
                mask,
                inverse,
                labels,
                max_length=args.max_length,
                min_new_tokens=args.min_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                early_stopping=args.num_beams > 1,
            )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for offset, (sample, prediction) in enumerate(zip(samples, predictions)):
                meta = sample["meta"]
                output_rows.append(
                    {
                        "example_id": meta["example_id"],
                        "sentence_id": meta["sentence_id"],
                        "task": meta["task"],
                        "subject": meta["subject"],
                        "reference": meta["reference"],
                        "prediction": prediction.strip(),
                        "seed": config["seed"],
                        "run_name": config["run_name"],
                        **({
                            "condition": args.reliance_condition,
                            "donor_example_id": (
                                dataset.inputs[plan[start + offset][args.reliance_condition]]["meta"]["example_id"]
                                if args.reliance_condition in ("mismatched", "wrong_subject") else ""
                            ),
                        } if plan is not None else {}),
                    }
                )
    output = args.output or args.run_dir / f"{args.phase}_predictions.csv"
    metrics = compute_metrics(
        [row["reference"] for row in output_rows], [row["prediction"] for row in output_rows]
    ).as_dict()
    write_csv(output, output_rows)
    metrics_output = output.with_suffix(".metrics.json")
    pending_metrics = metrics_output.with_suffix(".json.tmp") if signature else metrics_output
    write_json(
        pending_metrics,
        {
            **metrics,
            "run_name": config["run_name"],
            "seed": config["seed"],
            "checkpoint": str((args.run_dir / args.checkpoint).resolve()),
            "checkpoint_sha256": signature["checkpoint"] if signature else file_hash(args.run_dir / args.checkpoint),
            **({
                "reliance_signature": signature,
                "predictions_sha256": file_hash(output),
                "condition": args.reliance_condition,
                "eligible_examples": len(indices),
                "total_valid_examples": len(dataset),
                "excluded_examples": len(dataset) - len(indices),
                "perturbation_seed": args.perturbation_seed,
                "interpretation": "Paired inference-time interventions; wrong_subject preserves sentence content.",
            } if signature else {}),
            "elapsed_seconds": time.time() - started_at,
            "evaluation_phase": args.phase,
            "decoding": {
                "num_beams": args.num_beams,
                "max_length": args.max_length,
                "min_new_tokens": args.min_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "length_penalty": args.length_penalty,
            },
        },
    )
    if signature:
        pending_metrics.replace(metrics_output)
    print(metrics)


if __name__ == "__main__":
    main()
