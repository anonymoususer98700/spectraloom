from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODEL_PATH = Path(__file__).resolve().parents[2] / "EEG-To-text" / "model_decoding.py"
SPEC = importlib.util.spec_from_file_location("spectraloom_model_decoding", MODEL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_uniform_spectral_gate_is_exact_identity() -> None:
    gate = MODULE.SpectralBandAttention(n_bands=8, n_electrodes=3, uniform=True)
    x = torch.randn(2, 5, 24)
    torch.testing.assert_close(gate(x), x)


def test_learned_spectral_gate_preserves_shape_and_mean_scale() -> None:
    gate = MODULE.SpectralBandAttention(n_bands=8, n_electrodes=3, uniform=False)
    x = torch.randn(2, 5, 24)
    y = gate(x)
    assert y.shape == x.shape
    weights = torch.softmax(gate.band_query(gate.band_norm(x.view(2, 5, 8, 3))), dim=2)
    torch.testing.assert_close((weights * 8).mean(dim=2), torch.ones(2, 5, 1))


def test_cross_attention_bridge_zeros_padding_and_retains_length() -> None:
    bridge = MODULE.CrossAttentionBridge(12, 16, n_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 6, 12)
    padding = torch.tensor(
        [[False, False, False, False, True, True], [False, False, True, True, True, True]]
    )
    y = bridge(x, key_padding_mask=padding)
    assert y.shape == (2, 6, 16)
    torch.testing.assert_close(y[padding], torch.zeros_like(y[padding]))
    assert torch.count_nonzero(y[~padding]).item() > 0


def test_ablation_flags_replace_only_the_named_component() -> None:
    pretrained = torch.nn.Identity()
    no_sba = MODULE.EEGConformer(pretrained, in_feature=24, n_bands=8, n_electrodes=3)
    assert hasattr(no_sba, "band_attention")

    no_sba = MODULE.EEGConformer(
        pretrained, in_feature=24, n_bands=8, n_electrodes=3, ablate_sba=True
    )
    assert not hasattr(no_sba, "band_attention")
    assert hasattr(no_sba, "multi_scale_conv") and hasattr(no_sba, "bridge")

    no_multiscale = MODULE.EEGConformer(
        pretrained, in_feature=24, n_bands=8, n_electrodes=3, ablate_multiscale=True
    )
    assert hasattr(no_multiscale, "linear_conv")
    assert not hasattr(no_multiscale, "multi_scale_conv")
    assert hasattr(no_multiscale, "band_attention") and hasattr(no_multiscale, "bridge")

    no_cab = MODULE.EEGConformer(
        pretrained, in_feature=24, n_bands=8, n_electrodes=3, ablate_cab=True
    )
    assert hasattr(no_cab, "bridge_linear")
    assert not hasattr(no_cab, "bridge")
    assert hasattr(no_cab, "band_attention") and hasattr(no_cab, "multi_scale_conv")
