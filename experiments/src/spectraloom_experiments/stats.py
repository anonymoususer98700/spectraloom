from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Comparison:
    metric: str
    estimate_a_minus_b: float
    ci_low: float
    ci_high: float
    randomization_p: float
    clusters: int
    resamples: int
    higher_is_better: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _cluster_indices(cluster_ids: Sequence[str]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, cluster in enumerate(cluster_ids):
        grouped[str(cluster)].append(i)
    return [np.asarray(v, dtype=int) for v in grouped.values()]


def cluster_bootstrap_and_randomization(
    references: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    cluster_ids: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    metric_name: str,
    resamples: int = 10_000,
    seed: int = 2026,
    higher_is_better: bool = True,
) -> Comparison:
    n = len(references)
    if not (n and len(predictions_a) == len(predictions_b) == len(cluster_ids) == n):
        raise ValueError("All aligned inputs must have equal non-zero length")
    clusters = _cluster_indices(cluster_ids)
    if len(clusters) < 2:
        raise ValueError("At least two independent clusters are required")
    rng = np.random.default_rng(seed)
    estimate = metric(references, predictions_a) - metric(references, predictions_b)
    boot = np.empty(resamples, dtype=float)
    null = np.empty(resamples, dtype=float)
    refs = np.asarray(references, dtype=object)
    pa = np.asarray(predictions_a, dtype=object)
    pb = np.asarray(predictions_b, dtype=object)

    for iteration in range(resamples):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[j] for j in chosen])
        boot[iteration] = metric(refs[idx].tolist(), pa[idx].tolist()) - metric(
            refs[idx].tolist(), pb[idx].tolist()
        )

        swapped_a, swapped_b = pa.copy(), pb.copy()
        for cluster in clusters:
            if rng.random() < 0.5:
                swapped_a[cluster], swapped_b[cluster] = pb[cluster], pa[cluster]
        null[iteration] = metric(references, swapped_a.tolist()) - metric(
            references, swapped_b.tolist()
        )

    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    p = (1.0 + np.count_nonzero(np.abs(null) >= abs(estimate))) / (resamples + 1.0)
    return Comparison(
        metric=metric_name,
        estimate_a_minus_b=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        randomization_p=float(p),
        clusters=len(clusters),
        resamples=resamples,
        higher_is_better=higher_is_better,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p_values[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()

