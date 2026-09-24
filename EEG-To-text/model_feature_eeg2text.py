"""Feature-input EEG2Text-style baseline; not a reproduction of raw-EEG EEG2Text.

The original method uses raw temporal EEG, EEG pretraining, and anatomical
multi-view modeling. This matched-protocol adaptation instead accepts the
existing word-aligned eight-band features and trains from scratch on ZuCo 1.0.
"""

import torch
from torch import nn
from torch.nn import functional as F

from model_decoding import BrainTranslator


class EEG2TextFeatureAdapter(BrainTranslator):
    """Band/word CNN + Transformer encoder with the shared BART decoder."""

    def __init__(self, pretrained_layers, n_bands=8, n_electrodes=105):
        nn.Module.__init__(self)
        self.pretrained = pretrained_layers
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        self.temporal = nn.Conv2d(n_bands, 40, (1, 3), padding=(0, 1))
        self.spatial = nn.Conv2d(40, 40, (n_electrodes, 1))
        self.norm = nn.LayerNorm(40)
        self.dropout = nn.Dropout(0.5)
        layer = nn.TransformerEncoderLayer(
            40, 5, 2048, dropout=0.1, batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(
            layer, 6, enable_nested_tensor=False
        )
        self.fc1 = nn.Linear(40, pretrained_layers.config.d_model)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        padding = input_masks_invert.bool()
        x = input_embeddings_batch.masked_fill(padding.unsqueeze(-1), 0)
        x = x.reshape(*x.shape[:2], self.n_bands, self.n_electrodes)
        x = x.permute(0, 2, 3, 1)
        x = self.spatial(self.temporal(x)).squeeze(2).transpose(1, 2)
        x = self.dropout(F.elu(self.norm(x)))
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        x = self.additional_encoder(x, src_key_padding_mask=padding)
        return F.relu(self.fc1(x)).masked_fill(padding.unsqueeze(-1), 0)
