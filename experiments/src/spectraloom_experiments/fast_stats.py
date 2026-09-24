from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from .metrics import edit_distance, rouge_l_f1, token_f1, tokens


def _ngrams(sequence: Sequence[str], n: int) -> Counter:
    return Counter(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))


def bleu_stats(reference: str, prediction: str) -> np.ndarray:
    ref, pred = tokens(reference), tokens(prediction)
    values = []
    for n in range(1, 5):
        rc, pc = _ngrams(ref, n), _ngrams(pred, n)
        values.extend((sum((rc & pc).values()), sum(pc.values())))
    values.extend((len(ref), len(pred)))
    return np.asarray(values, dtype=float)


def bleu_from_stats(values: np.ndarray) -> float:
    ref_len, pred_len = values[-2:]
    if pred_len == 0:
        return 0.0
    precisions = [(values[2 * i] + 1.0) / (values[2 * i + 1] + 1.0) for i in range(4)]
    brevity = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1.0))
    return 100.0 * brevity * math.exp(sum(math.log(p) for p in precisions) / 4.0)


def _cluster_arrays(cluster_ids: Sequence[str], row_arrays: np.ndarray) -> np.ndarray:
    grouped = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        grouped[str(cluster)].append(index)
    return np.stack([row_arrays[indices].sum(axis=0) for indices in grouped.values()])


@dataclass(frozen=True)
class FastComparison:
    metric: str
    estimate_a_minus_b: float
    ci_low: float
    ci_high: float
    randomization_p: float
    clusters: int
    resamples: int
    higher_is_better: bool
    inference_implementation: str

    def as_dict(self) -> dict:
        return asdict(self)


def fast_cluster_comparison(
    references: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    cluster_ids: Sequence[str],
    metric: str,
    resamples: int,
    seed: int,
) -> FastComparison:
    if not (len(references) == len(predictions_a) == len(predictions_b) == len(cluster_ids)):
        raise ValueError("Aligned arrays differ in length")
    rng = np.random.default_rng(seed)
    if metric == "bootstrap_bleu4":
        a_rows = np.stack([bleu_stats(r, p) for r, p in zip(references, predictions_a)])
        b_rows = np.stack([bleu_stats(r, p) for r, p in zip(references, predictions_b)])
        a_clusters = _cluster_arrays(cluster_ids, a_rows)
        b_clusters = _cluster_arrays(cluster_ids, b_rows)
        score = bleu_from_stats
        implementation = "regex-tokenized BLEU-4; add-one n-gram smoothing; cluster resampling"
    elif metric in {"wer", "cer"}:
        if metric == "wer":
            ref_parts = [tokens(x) for x in references]
            a_parts = [tokens(x) for x in predictions_a]
            b_parts = [tokens(x) for x in predictions_b]
        else:
            ref_parts = [list(x) for x in references]
            a_parts = [list(x) for x in predictions_a]
            b_parts = [list(x) for x in predictions_b]
        a_rows = np.asarray(
            [[edit_distance(r, p), len(r)] for r, p in zip(ref_parts, a_parts)], dtype=float
        )
        b_rows = np.asarray(
            [[edit_distance(r, p), len(r)] for r, p in zip(ref_parts, b_parts)], dtype=float
        )
        a_clusters = _cluster_arrays(cluster_ids, a_rows)
        b_clusters = _cluster_arrays(cluster_ids, b_rows)
        score = lambda values: 100.0 * values[0] / max(values[1], 1.0)
        implementation = f"micro-averaged {metric.upper()}; cluster resampling"
    elif metric in {"rouge_l_f1", "token_f1"}:
        function = rouge_l_f1 if metric == "rouge_l_f1" else token_f1
        a_rows = np.asarray(
            [[100.0 * function(r, p), 1.0] for r, p in zip(references, predictions_a)]
        )
        b_rows = np.asarray(
            [[100.0 * function(r, p), 1.0] for r, p in zip(references, predictions_b)]
        )
        a_clusters = _cluster_arrays(cluster_ids, a_rows)
        b_clusters = _cluster_arrays(cluster_ids, b_rows)
        score = lambda values: values[0] / max(values[1], 1.0)
        implementation = f"macro-averaged {metric}; cluster resampling"
    else:
        raise ValueError(f"Unsupported fast paired metric {metric!r}")

    clusters = len(a_clusters)
    if clusters < 2:
        raise ValueError("At least two clusters are required")
    estimate = score(a_clusters.sum(axis=0)) - score(b_clusters.sum(axis=0))
    boot = np.empty(resamples)
    null = np.empty(resamples)
    for iteration in range(resamples):
        chosen = rng.integers(0, clusters, size=clusters)
        boot[iteration] = score(a_clusters[chosen].sum(axis=0)) - score(
            b_clusters[chosen].sum(axis=0)
        )
        swap = rng.random(clusters) < 0.5
        null_a = np.where(swap[:, None], b_clusters, a_clusters).sum(axis=0)
        null_b = np.where(swap[:, None], a_clusters, b_clusters).sum(axis=0)
        null[iteration] = score(null_a) - score(null_b)
    low, high = np.quantile(boot, [0.025, 0.975])
    p = (1 + np.count_nonzero(np.abs(null) >= abs(estimate))) / (resamples + 1)
    return FastComparison(
        metric=metric,
        estimate_a_minus_b=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        randomization_p=float(p),
        clusters=clusters,
        resamples=resamples,
        higher_is_better=metric not in {"wer", "cer"},
        inference_implementation=implementation,
    )

