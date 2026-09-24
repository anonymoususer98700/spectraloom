from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "EEG-To-text"))
sys.path.insert(0, str(REPO / "experiments" / "src"))

from model_feature_eeg2text import EEG2TextFeatureAdapter
from model_static_sba import StaticBandAttention
from spectraloom_experiments.reliance import CONDITIONS, paired_plan, perturb


class DummyPretrained(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=16)


def test_feature_input_adapter_shapes_and_padding():
    model = EEG2TextFeatureAdapter(DummyPretrained(), n_bands=2, n_electrodes=4).eval()
    eeg = torch.randn(2, 4, 8)
    padding = torch.tensor([[False, False, False, True], [False, False, True, True]])
    with torch.no_grad():
        output = model.addin_forward(eeg, padding)
    assert output.shape == (2, 4, 16)
    assert torch.count_nonzero(output[padding]) == 0


def test_static_sba_initially_preserves_scale_and_learns_band_weights():
    gate = StaticBandAttention(n_bands=2, n_electrodes=4)
    eeg = torch.randn(2, 3, 8)
    torch.testing.assert_close(gate(eeg), eeg)
    gate.logits.data[0] = 1.0
    assert not torch.allclose(gate(eeg), eeg)


def test_reliance_conditions_are_paired_and_deterministic():
    samples = []
    for sentence in ("s1", "s2"):
        for subject in ("a", "b"):
            index = len(samples)
            samples.append({
                "meta": {
                    "task": "zuco1_nr", "sentence_id": sentence,
                    "subject": subject, "example_id": str(index),
                },
                "seq_len": 3,
                "input_embeddings": torch.full((4, 8), float(index)),
                "input_attn_mask": torch.tensor([1, 1, 1, 0]),
            })
    plan = paired_plan(samples, 2026)
    assert len(plan) == len(samples)
    for entry in plan:
        source = samples[entry["index"]]["meta"]
        assert samples[entry["mismatched"]]["meta"]["sentence_id"] != source["sentence_id"]
        assert samples[entry["wrong_subject"]]["meta"]["sentence_id"] == source["sentence_id"]
        assert samples[entry["wrong_subject"]]["meta"]["subject"] != source["subject"]
        for condition in CONDITIONS:
            first = perturb(samples, entry, condition, 2026)
            second = perturb(samples, entry, condition, 2026)
            for left, right in zip(first, second):
                torch.testing.assert_close(left, right)
        prior_eeg, prior_mask, _ = perturb(samples, entry, "language_prior", 2026)
        assert torch.count_nonzero(prior_eeg) == 0
        assert torch.all(prior_mask == 1)
