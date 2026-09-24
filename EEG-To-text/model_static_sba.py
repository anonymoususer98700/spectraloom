"""Input-independent spectral-band gate for a SpectraLoom SBA control."""

import torch
from torch import nn

from model_decoding import EEGConformer


class StaticBandAttention(nn.Module):
    def __init__(self, n_bands=8, n_electrodes=105):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        self.logits = nn.Parameter(torch.zeros(n_bands))

    def forward(self, x):
        bands = x.reshape(*x.shape[:2], self.n_bands, self.n_electrodes)
        weights = self.logits.softmax(0) * self.n_bands
        return (bands * weights.view(1, 1, -1, 1)).reshape_as(x)


class EEGConformerStaticSBA(EEGConformer):
    """Original model with only dynamic SBA replaced by eight fixed logits."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("ablate_sba"):
            raise ValueError("Static SBA cannot be combined with SBA ablations")
        super().__init__(*args, **kwargs)
        self.band_attention = StaticBandAttention(
            kwargs.get("n_bands", 8), kwargs.get("n_electrodes", 105)
        )
