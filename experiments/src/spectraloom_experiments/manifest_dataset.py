from __future__ import annotations

import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .protocols import load_task


DEFAULT_BANDS = ("_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2")


def read_manifest(path: str | Path, phase: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == phase]
    if not rows:
        raise ValueError(f"Manifest has no rows for phase {phase!r}")
    return rows


def raw_vectors(sent_obj: dict, eeg_type: str, bands: list[str]) -> np.ndarray | None:
    vectors = []
    for word in sent_obj.get("word") or []:
        try:
            parts = [word["word_level_EEG"][eeg_type][eeg_type + band] for band in bands]
            vector = np.concatenate(parts).astype(np.float32, copy=False)
        except (KeyError, TypeError, ValueError):
            return None
        if vector.shape != (105 * len(bands),) or not np.isfinite(vector).all():
            return None
        vectors.append(vector)
    if not vectors:
        return None
    return np.stack(vectors)


def compute_train_statistics(
    rows: list[dict], datasets: dict[str, dict], eeg_type: str, bands: list[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(105 * len(bands), dtype=np.float64)
    squared = np.zeros_like(total)
    count = 0
    for row in tqdm(rows, desc="Train normalization", unit="row", leave=False):
        sent_obj = datasets[row["task"]][row["subject"]][int(row["source_index"])]
        vectors = raw_vectors(sent_obj, eeg_type, bands)
        if vectors is None:
            continue
        total += vectors.sum(axis=0)
        squared += np.square(vectors, dtype=np.float64).sum(axis=0)
        count += vectors.shape[0]
    if count < 2:
        raise ValueError("Insufficient valid training vectors for feature statistics")
    mean = total / count
    variance = np.maximum(squared / count - mean**2, 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count


class ManifestZuCoDataset:
    def __init__(
        self,
        manifest: str | Path,
        phase: str,
        pickle_dir: str | Path,
        tokenizer,
        eeg_type: str = "GD",
        bands: list[str] | None = None,
        max_length: int = 56,
        normalization: str = "train_feature",
        train_mean: np.ndarray | None = None,
        train_std: np.ndarray | None = None,
    ) -> None:
        import torch

        started = time.perf_counter()
        self.rows = read_manifest(manifest, phase)
        self.bands = list(bands or DEFAULT_BANDS)
        self.eeg_type = eeg_type
        self.max_length = max_length
        self.normalization = normalization
        tasks = sorted({row["task"] for row in self.rows})
        self.datasets = {task: load_task(pickle_dir, task) for task in tasks}
        self.inputs = []
        for row in tqdm(self.rows, desc=f"Materialize {phase}", unit="row", leave=False):
            sent_obj = self.datasets[row["task"]][row["subject"]][int(row["source_index"])]
            vectors = raw_vectors(sent_obj, eeg_type, self.bands)
            if vectors is None:
                continue
            vectors = vectors[:max_length]
            if normalization == "per_word":
                means = vectors.mean(axis=1, keepdims=True)
                stds = vectors.std(axis=1, keepdims=True)
                vectors = (vectors - means) / np.maximum(stds, 1e-8)
            elif normalization == "train_feature":
                if train_mean is None or train_std is None:
                    raise ValueError("train_feature normalization requires train_mean and train_std")
                vectors = (vectors - train_mean) / np.maximum(train_std, 1e-8)
            elif normalization != "none":
                raise ValueError("normalization must be train_feature, per_word, or none")
            length = vectors.shape[0]
            padded = np.zeros((max_length, vectors.shape[1]), dtype=np.float32)
            padded[:length] = vectors
            input_mask = np.zeros(max_length, dtype=np.float32)
            input_mask[:length] = 1
            target = tokenizer(
                row["reference"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.inputs.append(
                {
                    "input_embeddings": torch.from_numpy(padded),
                    "seq_len": length,
                    "input_attn_mask": torch.from_numpy(input_mask),
                    "input_attn_mask_invert": torch.from_numpy(1 - input_mask).bool(),
                    "target_ids": target["input_ids"][0],
                    "target_mask": target["attention_mask"][0],
                    "meta": row,
                }
            )
        # Samples are materialized; retaining multi-gigabyte pickle dictionaries
        # would multiply memory use across train/dev datasets.
        self.datasets.clear()
        if not self.inputs:
            raise ValueError(f"No valid {phase} samples after EEG validation")
        self.loading_seconds = time.perf_counter() - started
        print(
            f"Prepared {phase}: {len(self.inputs)}/{len(self.rows)} valid rows "
            f"in {self.loading_seconds:.2f}s",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        sample = self.inputs[index]
        return (
            sample["input_embeddings"],
            sample["seq_len"],
            sample["input_attn_mask"],
            sample["input_attn_mask_invert"],
            sample["target_ids"],
            sample["target_mask"],
        )
