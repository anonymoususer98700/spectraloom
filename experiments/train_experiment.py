#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from spectraloom_experiments.io import write_json
from spectraloom_experiments.legacy_adapter import instantiate_legacy_model, seed_everything, tokenizer_name
from spectraloom_experiments.manifest_dataset import (
    DEFAULT_BANDS,
    ManifestZuCoDataset,
    compute_train_statistics,
    read_manifest,
)
from spectraloom_experiments.protocols import load_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest-driven SpectraLoom extension training.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("EEGConformer", "EEGConformerStaticSBA", "EEG2TextFeatureAdapter", "BrainTranslator", "R1Translator", "T5Translator", "PegasusTranslator"),
        default="EEGConformer",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "training")
    parser.add_argument("--pickle-dir", type=Path, default=REPO / "EEG-To-text" / "Data" / "pickle_file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--custom-lr", type=float, default=2e-5)
    parser.add_argument("--pretrained-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=56)
    parser.add_argument("--normalization", choices=("train_feature", "per_word", "none"), default="train_feature")
    parser.add_argument(
        "--train-control",
        choices=("real", "zero_values_original_mask", "zero_values_full_mask", "gaussian_original_mask"),
        default="real",
        help="Train matched side-channel/language-prior baselines with the same capacity and optimizer.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16",
        help="Mixed-precision dtype. BF16 is the stable default on RTX 50-series GPUs.",
    )
    parser.add_argument(
        "--save-last", action="store_true",
        help="Also retain last.pt. By default only best.pt is kept to limit disk use.",
    )
    parser.add_argument("--ablate-sba", action="store_true")
    parser.add_argument("--uniform-sba", action="store_true")
    parser.add_argument("--ablate-multiscale", action="store_true")
    parser.add_argument("--ablate-cab", action="store_true")
    parser.add_argument("--conv-kernels", default="3,5,7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    if args.ablate_sba and args.uniform_sba:
        raise ValueError("Choose either no SBA or uniform SBA, not both")
    started_at = time.time()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_id = tokenizer_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    bands = list(DEFAULT_BANDS)

    train_rows = read_manifest(args.manifest, "train")
    tasks = sorted({row["task"] for row in train_rows})
    loader_started = time.perf_counter()
    datasets = {task: load_task(args.pickle_dir, task) for task in tasks}
    train_mean = train_std = None
    stats_count = 0
    if args.normalization == "train_feature":
        train_mean, train_std, stats_count = compute_train_statistics(
            train_rows, datasets, "GD", bands
        )
        np.savez(run_dir / "normalization.npz", mean=train_mean, std=train_std, count=stats_count)
    del datasets

    train_set = ManifestZuCoDataset(
        args.manifest, "train", args.pickle_dir, tokenizer, bands=bands,
        max_length=args.max_length, normalization=args.normalization,
        train_mean=train_mean, train_std=train_std,
    )
    dev_set = ManifestZuCoDataset(
        args.manifest, "dev", args.pickle_dir, tokenizer, bands=bands,
        max_length=args.max_length, normalization=args.normalization,
        train_mean=train_mean, train_std=train_std,
    )
    data_preparation_seconds = time.perf_counter() - loader_started
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    config = {
        "model_name": args.model,
        "eeg_bands": bands,
        "ablate_sba": "uniform" if args.uniform_sba else args.ablate_sba,
        "ablate_multiscale": args.ablate_multiscale,
        "ablate_cab": args.ablate_cab,
        "conv_kernels": args.conv_kernels,
        "dropout": 0.3,
    }
    model = instantiate_legacy_model(REPO, config, device)
    unfreeze = ["shared", "embed_positions", "embed_tokens", "layernorm_embedding"]
    unfreeze += [f"encoder.layers.{i}." for i in range(4)]
    unfreeze += [f"decoder.layers.{i}." for i in range(4)]
    unfreeze += [f"encoder.block.{i}." for i in range(4)]
    unfreeze += [f"decoder.block.{i}." for i in range(4)]
    pretrained_params, custom_params = [], []
    for name, parameter in model.named_parameters():
        if "pretrained" in name:
            parameter.requires_grad = any(key in name for key in unfreeze)
            if parameter.requires_grad:
                pretrained_params.append(parameter)
        else:
            parameter.requires_grad = True
            custom_params.append(parameter)
    optimizer = AdamW(
        [
            {"params": custom_params, "lr": args.custom_lr},
            {"params": pretrained_params, "lr": args.pretrained_lr},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_steps = math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    warmup_steps = max(1, int(optimizer_steps * args.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, schedule)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    run_config = vars(args).copy()
    run_config.update(
        {
            "manifest": str(args.manifest.resolve()),
            "pickle_dir": str(args.pickle_dir.resolve()),
            "device_used": str(device),
            "train_examples": len(train_set),
            "dev_examples": len(dev_set),
            "train_dataset_loading_seconds": train_set.loading_seconds,
            "dev_dataset_loading_seconds": dev_set.loading_seconds,
            "data_preparation_seconds": data_preparation_seconds,
            "normalization_statistics_word_vectors": stats_count,
            "optimizer_steps": optimizer_steps,
            "warmup_steps": warmup_steps,
            "loss": "cross_entropy",
            "scheduled_sampling": False,
            "amp_enabled": amp_enabled,
            "amp_dtype_used": str(amp_dtype) if amp_enabled else None,
            "gradient_scaler_enabled": scaler_enabled,
            "tokenizer_id": tokenizer_id,
            "pretrained_model_id": tokenizer_id,
            "random_seed": args.seed,
            "numpy_seed": args.seed,
            "torch_seed": args.seed,
            "dataloader_seed": args.seed,
            "python_hash_seed": str(args.seed),
            "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "model_source_sha256": hashlib.sha256((REPO / "EEG-To-text" / "model_decoding.py").read_bytes()).hexdigest(),
            "variant_source_sha256": (
                hashlib.sha256((REPO / "EEG-To-text" / (
                    "model_static_sba.py" if args.model == "EEGConformerStaticSBA"
                    else "model_feature_eeg2text.py"
                )).read_bytes()).hexdigest()
                if args.model in {"EEGConformerStaticSBA", "EEG2TextFeatureAdapter"} else None
            ),
            "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started_unix_time": started_at,
        }
    )
    for key, value in list(run_config.items()):
        if isinstance(value, Path):
            run_config[key] = str(value)
    write_json(run_dir / "config.json", run_config)

    def batch_loss(batch, training: bool):
        embeddings, _, mask, inverse, targets, target_mask = batch
        embeddings = embeddings.to(device, non_blocking=True).float()
        mask = mask.to(device, non_blocking=True)
        inverse = inverse.to(device, non_blocking=True)
        if args.train_control == "zero_values_original_mask":
            embeddings.zero_()
        elif args.train_control == "zero_values_full_mask":
            embeddings.zero_(); mask.fill_(1); inverse.zero_()
        elif args.train_control == "gaussian_original_mask":
            embeddings = torch.randn_like(embeddings) * mask.unsqueeze(-1)
        labels = targets.to(device, non_blocking=True)
        labels = labels.masked_fill(target_mask.to(device) == 0, -100)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(embeddings, mask, inverse, labels, tf_ratio=1.0)
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=args.label_smoothing,
            )
        return loss

    best = float("inf")
    stale = 0
    global_step = 0
    log_path = run_dir / "history.jsonl"
    resume_path = run_dir / "resume.pt"
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        best = float(resume["best"])
        stale = int(resume["stale"])
        global_step = int(resume["global_step"])
        start_epoch = int(resume["epoch"]) + 1
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "cuda" and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        generator.set_state(resume["dataloader_rng"])
        print(f"Resuming {args.run_name} at epoch {start_epoch}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} train",
            unit="batch", dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(train_progress, 1):
            loss = batch_loss(batch, True)
            scaler.scale(loss / args.gradient_accumulation).backward()
            train_total += float(loss.detach())
            train_progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_was_skipped = scaler_enabled and scaler.get_scale() < scale_before
                optimizer.zero_grad(set_to_none=True)
                if not step_was_skipped:
                    scheduler.step()
                    global_step += 1
        model.eval()
        dev_total = 0.0
        with torch.inference_mode():
            for batch in tqdm(
                dev_loader, desc=f"Epoch {epoch}/{args.epochs} dev",
                unit="batch", dynamic_ncols=True,
            ):
                dev_total += float(batch_loss(batch, False))
        record = {
            "epoch": epoch,
            "train_loss": train_total / len(train_loader),
            "dev_loss": dev_total / len(dev_loader),
            "global_step": global_step,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "unix_time": time.time(),
            "elapsed_seconds": time.time() - epoch_started,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else None
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(record)
        if args.save_last:
            torch.save(model.state_dict(), run_dir / "last.pt")
        if record["dev_loss"] < best:
            best = record["dev_loss"]; stale = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best,
                "stale": stale,
                "global_step": global_step,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "dataloader_rng": generator.get_state(),
            },
            resume_path,
        )
        if stale >= args.patience:
            break
    checkpoint = run_dir / "best.pt"
    write_json(
        run_dir / "complete.json",
        {
            "best_dev_loss": best,
            "epochs_completed": epoch,
            "elapsed_seconds": time.time() - started_at,
            "best_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    resume_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
