from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Sequence


TOKEN_RE = re.compile(r"[\w]+(?:'[\w]+)?|[^\w\s]", re.UNICODE)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (item_a != item_b))
            )
        previous = current
    return previous[-1]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for item_a in a:
        current = [0]
        for j, item_b in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if item_a == item_b else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref, pred = tokens(reference), tokens(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_length(ref, pred)
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def token_f1(reference: str, prediction: str) -> float:
    ref, pred = Counter(tokens(reference)), Counter(tokens(prediction))
    overlap = sum((ref & pred).values())
    if not ref or not pred or overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def _ngram_counts(sequence: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))


def corpus_bleu(references: Sequence[str], predictions: Sequence[str], max_order: int = 4) -> float:
    """Canonical BLEU via sacrebleu when installed; a deterministic fallback otherwise."""
    try:
        import sacrebleu

        metric = sacrebleu.metrics.BLEU(max_ngram_order=max_order, effective_order=True)
        return float(metric.corpus_score(list(predictions), [list(references)]).score)
    except ImportError:
        matches = [0] * max_order
        totals = [0] * max_order
        ref_len = pred_len = 0
        for reference, prediction in zip(references, predictions):
            ref, pred = tokens(reference), tokens(prediction)
            ref_len += len(ref)
            pred_len += len(pred)
            for n in range(1, max_order + 1):
                ref_counts, pred_counts = _ngram_counts(ref, n), _ngram_counts(pred, n)
                matches[n - 1] += sum((ref_counts & pred_counts).values())
                totals[n - 1] += sum(pred_counts.values())
        if pred_len == 0:
            return 0.0
        precisions = [(matches[i] + 1.0) / (totals[i] + 1.0) for i in range(max_order)]
        bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))
        return 100.0 * bp * math.exp(sum(math.log(p) for p in precisions) / max_order)


def chrf(references: Sequence[str], predictions: Sequence[str]) -> float:
    try:
        import sacrebleu

        return float(sacrebleu.corpus_chrf(list(predictions), [list(references)]).score)
    except ImportError:
        return 100.0 * mean(token_f1(r, p) for r, p in zip(references, predictions))


def bleu_signature() -> str:
    try:
        import sacrebleu

        metric = sacrebleu.metrics.BLEU()
        metric.corpus_score(["x"], [["x"]])
        return str(metric.get_signature())
    except ImportError:
        return "fallback:lowercase-regex-tokenization:add-one-smoothing"


@dataclass(frozen=True)
class MetricBundle:
    n: int
    unique_references: int
    corpus_bleu: float
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    chrf: float
    rouge_l_f1: float
    token_f1: float
    wer: float
    cer: float
    exact_match: float
    empty_prediction_rate: float
    unique_prediction_rate: float
    prediction_reference_length_ratio: float
    bleu_signature: str

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(references: Sequence[str], predictions: Sequence[str]) -> MetricBundle:
    if len(references) != len(predictions) or not references:
        raise ValueError("references and predictions must be non-empty and have equal length")
    ref_tok = [tokens(x) for x in references]
    pred_tok = [tokens(x) for x in predictions]
    word_errors = sum(edit_distance(r, p) for r, p in zip(ref_tok, pred_tok))
    ref_words = sum(len(x) for x in ref_tok)
    char_errors = sum(edit_distance(list(r), list(p)) for r, p in zip(references, predictions))
    ref_chars = sum(len(r) for r in references)
    return MetricBundle(
        n=len(references),
        unique_references=len(set(references)),
        corpus_bleu=corpus_bleu(references, predictions),
        bleu_1=corpus_bleu(references, predictions, 1),
        bleu_2=corpus_bleu(references, predictions, 2),
        bleu_3=corpus_bleu(references, predictions, 3),
        bleu_4=corpus_bleu(references, predictions, 4),
        chrf=chrf(references, predictions),
        rouge_l_f1=100.0 * mean(rouge_l_f1(r, p) for r, p in zip(references, predictions)),
        token_f1=100.0 * mean(token_f1(r, p) for r, p in zip(references, predictions)),
        wer=100.0 * word_errors / max(ref_words, 1),
        cer=100.0 * char_errors / max(ref_chars, 1),
        exact_match=100.0 * mean(r == p for r, p in zip(references, predictions)),
        empty_prediction_rate=100.0 * mean(not p.strip() for p in predictions),
        unique_prediction_rate=100.0 * len(set(predictions)) / len(predictions),
        prediction_reference_length_ratio=sum(len(x) for x in pred_tok) / max(ref_words, 1),
        bleu_signature=bleu_signature(),
    )


def scalar_metric(name: str, references: Sequence[str], predictions: Sequence[str]) -> float:
    if name == "corpus_bleu":
        return corpus_bleu(references, predictions)
    if name == "chrf":
        return chrf(references, predictions)
    if name == "rouge_l_f1":
        return 100.0 * mean(rouge_l_f1(r, p) for r, p in zip(references, predictions))
    if name == "token_f1":
        return 100.0 * mean(token_f1(r, p) for r, p in zip(references, predictions))
    if name in {"wer", "cer"}:
        if name == "wer":
            refs = [tokens(x) for x in references]
            preds = [tokens(x) for x in predictions]
        else:
            refs = [list(x) for x in references]
            preds = [list(x) for x in predictions]
        errors = sum(edit_distance(r, p) for r, p in zip(refs, preds))
        denominator = sum(len(r) for r in refs)
        return 100.0 * errors / max(denominator, 1)
    raise ValueError(f"Unknown scalar metric: {name}")
