"""Paired, deterministic inference-time EEG interventions."""

import hashlib
import random
from collections import defaultdict

import torch


CONDITIONS = (
    "real", "gaussian", "language_prior", "mismatched", "shuffled", "wrong_subject"
)


def keyed_seed(seed, example_id, condition):
    key = f"{seed}:{example_id}:{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def paired_plan(samples, seed):
    """Use the same eligible trials for all six conditions.

    Mismatched EEG keeps task and event length but changes sentence. Wrong-
    subject EEG keeps the sentence and event length but changes subject; it is
    an invariance check, not a control expected to lower performance.
    """
    by_length, by_sentence = defaultdict(list), defaultdict(list)
    for index, sample in enumerate(samples):
        meta = sample["meta"]
        by_length[(meta["task"], sample["seq_len"])].append(index)
        by_sentence[(meta["task"], meta["sentence_id"], sample["seq_len"])].append(index)
    plan = []
    for index, sample in enumerate(samples):
        meta = sample["meta"]
        different_sentence = [
            donor for donor in by_length[(meta["task"], sample["seq_len"])]
            if samples[donor]["meta"]["sentence_id"] != meta["sentence_id"]
        ]
        different_subject = [
            donor for donor in by_sentence[(meta["task"], meta["sentence_id"], sample["seq_len"])]
            if samples[donor]["meta"]["subject"] != meta["subject"]
        ]
        if not different_sentence or not different_subject or sample["seq_len"] < 2:
            continue
        rng = random.Random(keyed_seed(seed, meta["example_id"], "donors"))
        plan.append({
            "index": index,
            "mismatched": rng.choice(different_sentence),
            "wrong_subject": rng.choice(different_subject),
        })
    if not plan:
        raise ValueError("No common eligible trials for paired reliance controls")
    return plan


def perturb(samples, entry, condition, seed):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    sample = samples[entry["index"]]
    eeg = sample["input_embeddings"].clone()
    mask = sample["input_attn_mask"].clone()
    length = sample["seq_len"]
    generator = torch.Generator().manual_seed(
        keyed_seed(seed, sample["meta"]["example_id"], condition)
    )
    if condition in ("mismatched", "wrong_subject"):
        eeg = samples[entry[condition]]["input_embeddings"].clone()
    elif condition == "gaussian":
        # Unit Gaussian is meaningful only in train-feature-normalized space.
        eeg = torch.randn(eeg.shape, generator=generator) * mask.unsqueeze(-1)
    elif condition == "language_prior":
        eeg.zero_()
        mask.fill_(1)  # Also removes the true EEG event-length side channel.
    elif condition == "shuffled":
        order = torch.randperm(length, generator=generator)
        if torch.equal(order, torch.arange(length)):
            order = order.roll(1)
        eeg[:length] = eeg[:length][order]
    return eeg, mask, ~mask.bool()
