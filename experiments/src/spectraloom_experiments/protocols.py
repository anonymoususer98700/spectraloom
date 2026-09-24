from __future__ import annotations

import hashlib
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from .io import canonical_text, example_id, sentence_id


TASK_FILES = {
    "zuco1_sr": "task1-SR-dataset.pickle",
    "zuco1_nr": "task2-NR-dataset.pickle",
    "zuco1_tsr": "task3-TSR-dataset.pickle",
    "zuco2_tsr": "task2-TSR-2.0-dataset.pickle",
}


def load_task(pickle_dir: str | Path, task: str) -> dict:
    if task not in TASK_FILES:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASK_FILES)}")
    path = Path(pickle_dir) / TASK_FILES[task]
    if not path.exists():
        raise FileNotFoundError(path)
    started = time.perf_counter()
    size = path.stat().st_size
    with path.open("rb") as raw:
        with tqdm.wrapattr(
            raw, "read", total=size, desc=f"Load {task}", unit="B",
            unit_scale=True, unit_divisor=1024, leave=False,
        ) as handle:
            value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected subject dictionary in {path}")
    print(
        f"Loaded {task} ({path.name}, {size / 1024**2:.1f} MiB) "
        f"in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return value


def iter_records(dataset: dict, task: str) -> Iterable[dict]:
    occurrences: Counter[tuple[str, str]] = Counter()
    for subject in sorted(dataset):
        samples = dataset[subject]
        for source_index, sent_obj in enumerate(samples):
            if sent_obj is None:
                continue
            reference = canonical_text(str(sent_obj.get("content", "")))
            if not reference:
                continue
            key = (subject, reference)
            occurrence = occurrences[key]
            occurrences[key] += 1
            words = sent_obj.get("word") or []
            yield {
                "example_id": example_id(task, subject, reference, occurrence),
                "sentence_id": sentence_id(task, reference),
                "task": task,
                "subject": subject,
                "source_index": source_index,
                "occurrence": occurrence,
                "reference": reference,
                "word_count": len(words),
                "character_count": len(reference),
                "sent_obj": sent_obj,
            }


def _rank(task: str, reference: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{task}\0{reference}".encode("utf-8")).hexdigest()


def sentence_partitions(records: list[dict], seed: int = 2026) -> dict[str, str]:
    references = {row["reference"] for row in records}
    result: dict[str, str] = {}
    ordered = sorted(references, key=lambda text: _rank("all_tasks", text, seed))
    n = len(ordered)
    train_end = int(0.8 * n)
    dev_end = train_end + int(0.1 * n)
    for index, reference in enumerate(ordered):
        phase = "train" if index < train_end else "dev" if index < dev_end else "test"
        result[sentence_id("all_tasks", reference)] = phase
    return result


def build_manifest(
    datasets: dict[str, dict],
    protocol: str = "sentence_group",
    seed: int = 2026,
    held_out_subject: str | None = None,
) -> list[dict]:
    records = [
        row
        for task, dataset in datasets.items()
        for row in iter_records(dataset, task)
    ]
    partitions = sentence_partitions(records, seed)
    manifest: list[dict] = []
    all_subjects = {row["subject"] for row in records}
    if protocol == "loso_unseen_sentence" and not held_out_subject:
        raise ValueError("--held-out-subject is required for loso_unseen_sentence")
    if held_out_subject and held_out_subject not in all_subjects:
        raise ValueError(
            f"Held-out subject {held_out_subject!r} absent; available: {sorted(all_subjects)}"
        )

    for row in records:
        sentence_phase = partitions[row["sentence_id"]]
        if protocol == "sentence_group":
            phase = sentence_phase
        elif protocol == "loso_unseen_sentence":
            if sentence_phase == "test":
                phase = "test" if row["subject"] == held_out_subject else "excluded"
            elif row["subject"] == held_out_subject:
                phase = "excluded"
            else:
                phase = sentence_phase
        else:
            raise ValueError("protocol must be sentence_group or loso_unseen_sentence")
        manifest.append(
            {
                key: value
                for key, value in row.items()
                if key != "sent_obj"
            }
            | {
                "phase": phase,
                "protocol": protocol,
                "split_seed": seed,
                "held_out_subject": held_out_subject or "",
            }
        )
    return manifest


def validate_manifest(rows: list[dict]) -> dict:
    phases_by_sentence: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    subjects = set()
    for row in rows:
        counts[row["phase"]] += 1
        subjects.add(row["subject"])
        if row["phase"] != "excluded":
            phases_by_sentence[row["sentence_id"]].add(row["phase"])
    overlaps = {sid: sorted(phases) for sid, phases in phases_by_sentence.items() if len(phases) > 1}
    if overlaps:
        raise ValueError(f"Sentence leakage across phases: {len(overlaps)} sentence IDs")
    return {
        "rows_by_phase": dict(counts),
        "unique_sentences": len(phases_by_sentence),
        "subjects": sorted(subjects),
        "sentence_overlap_count": 0,
    }
