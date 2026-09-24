#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from spectraloom_experiments.io import write_csv, write_json
from spectraloom_experiments.legacy_adapter import (
    instantiate_legacy_model,
    load_checkpoint,
    seed_everything,
    tokenizer_name,
)
from spectraloom_experiments.manifest_dataset import DEFAULT_BANDS, ManifestZuCoDataset
from spectraloom_experiments.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate next-token predictions with gold-prefix teacher forcing."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--phase", choices=("dev", "test", "zero_shot"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    started_at = time.time()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_id", tokenizer_name(config["model"]))
    )
    mean = std = None
    if config["normalization"] == "train_feature":
        stats = np.load(args.run_dir / "normalization.npz")
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
    checkpoint = args.run_dir / args.checkpoint
    model = instantiate_legacy_model(REPO, model_config, device)
    load_checkpoint(model, checkpoint)
    model.eval()

    rows: list[dict] = []
    total_nll = 0.0
    correct_tokens = 0
    token_count = 0
    starts = range(0, len(dataset), args.batch_size)
    amp_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for start in tqdm(
            starts,
            total=len(starts),
            desc=f"Teacher-forced {args.phase}",
            unit="batch",
            dynamic_ncols=True,
        ):
            samples = dataset.inputs[start : start + args.batch_size]
            embeddings = torch.stack([s["input_embeddings"] for s in samples]).to(device).float()
            mask = torch.stack([s["input_attn_mask"] for s in samples]).to(device)
            inverse = torch.stack([s["input_attn_mask_invert"] for s in samples]).to(device)
            targets = torch.stack([s["target_ids"] for s in samples]).to(device)
            target_mask = torch.stack([s["target_mask"] for s in samples]).to(device).bool()
            labels = targets.masked_fill(~target_mask, -100)

            control = config.get("train_control", "real")
            if control == "zero_values_original_mask":
                embeddings.zero_()
            elif control == "zero_values_full_mask":
                embeddings.zero_(); mask.fill_(1); inverse.zero_()
            elif control == "gaussian_original_mask":
                embeddings = torch.randn_like(embeddings) * mask.unsqueeze(-1)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=amp_enabled,
            ):
                output = model(embeddings, mask, inverse, labels, tf_ratio=1.0)
            logits = output.logits.float()
            predicted = logits.argmax(dim=-1)
            total_nll += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            )
            valid = labels.ne(-100)
            correct_tokens += int((predicted.eq(labels) & valid).sum())
            token_count += int(valid.sum())

            for sample, predicted_ids, valid_positions in zip(samples, predicted, valid):
                prediction = tokenizer.decode(
                    predicted_ids[valid_positions].tolist(), skip_special_tokens=True
                ).strip()
                meta = sample["meta"]
                rows.append(
                    {
                        "example_id": meta["example_id"],
                        "sentence_id": meta["sentence_id"],
                        "task": meta["task"],
                        "subject": meta["subject"],
                        "reference": meta["reference"],
                        "prediction": prediction,
                        "seed": config["seed"],
                        "run_name": config["run_name"],
                    }
                )

    output_path = args.output or args.run_dir / f"teacher_forced_{args.phase}_predictions.csv"
    metrics = compute_metrics(
        [row["reference"] for row in rows], [row["prediction"] for row in rows]
    ).as_dict()
    mean_nll = total_nll / token_count
    write_csv(output_path, rows)
    write_json(
        output_path.with_suffix(".metrics.json"),
        {
            **metrics,
            "teacher_forced_token_accuracy": 100.0 * correct_tokens / token_count,
            "teacher_forced_nll": mean_nll,
            "teacher_forced_perplexity": math.exp(min(mean_nll, 50.0)),
            "run_name": config["run_name"],
            "seed": config["seed"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "elapsed_seconds": time.time() - started_at,
            "evaluation_phase": args.phase,
            "evaluation_mode": "teacher_forced_gold_prefix_argmax",
            "n": len(rows),
        },
    )
    print({**metrics, "token_accuracy": 100.0 * correct_tokens / token_count, "nll": mean_nll})


if __name__ == "__main__":
    main()
