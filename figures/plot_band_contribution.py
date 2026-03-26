import matplotlib.pyplot as plt
import numpy as np
import torch
from model_decoding import EEGConformer
from Data import ZuCo_dataset
import json
import os

# Load config and checkpoint paths (edit as needed)
CONFIG_PATH = './config/decoding/task1v1_task2v1_task3v1_finetune_EEGConformer_skipstep1_b32_20_30_0.0001_2e-05_unique_sent_EEG.json'
CHECKPOINT_PATH = './checkpoints/decoding/best/task1v1_task2v1_task3v1_finetune_EEGConformer_skipstep1_b32_20_30_0.0001_2e-05_unique_sent_EEG.pt'

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Load model
def load_model(config, checkpoint_path, device):
    from transformers import BartForConditionalGeneration
    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large', torch_dtype=torch.float32)
    pretrained = pretrained.float()
    model = EEGConformer(
        pretrained,
        in_feature=105*len(config['eeg_bands']),
        decoder_embedding_size=pretrained.config.d_model,
        n_bands=len(config['eeg_bands']),
        n_electrodes=105,
        conv_channels=105*len(config['eeg_bands']),
        rnn_hidden_size=512,
        num_rnn_layers=3,
        n_bridge_heads=8,
        dropout=0.2,
        ablate_sba=config.get('ablate_sba', False),
        ablate_multiscale=config.get('ablate_multiscale', False),
        ablate_cab=config.get('ablate_cab', False),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# Extract and plot band attention weights
def plot_band_attention(model, eeg_sample, bands, save_path='figures/band_contribution.png'):
    # Forward through input norm and band attention only
    x = model.input_norm(eeg_sample)
    attn_module = model.band_attention
    B, L, _ = x.shape
    x_bands = x.view(B, L, attn_module.n_bands, attn_module.n_electrodes)
    attn_logits = attn_module.band_query(attn_module.band_norm(x_bands))
    attn_weights = torch.softmax(attn_logits, dim=2)  # (B, L, n_bands, 1)
    attn_weights = attn_weights.squeeze(-1).detach().cpu().numpy()  # (B, L, n_bands)
    # Average over all words and batch
    mean_band_weights = attn_weights.mean(axis=(0,1))
    plt.figure(figsize=(8,4))
    plt.bar(bands, mean_band_weights)
    plt.ylabel('Mean Attention Weight')
    plt.xlabel('EEG Frequency Band')
    plt.title('EEG Frequency Band Contribution to Model')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Band contribution figure saved to {save_path}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(CONFIG_PATH)
    model = load_model(config, CHECKPOINT_PATH, device)
    # Load a batch of EEG data (use a few samples for averaging)
    PICKLE_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'pickle_file')
    whole_dataset_dicts = []
    if 'task1' in config['task_name']:
        with open(os.path.join(PICKLE_DIR, 'task1-SR-dataset.pickle'), 'rb') as h:
            whole_dataset_dicts.append(torch.load(h, map_location=device))
    if 'task2' in config['task_name']:
        with open(os.path.join(PICKLE_DIR, 'task2-NR-dataset.pickle'), 'rb') as h:
            whole_dataset_dicts.append(torch.load(h, map_location=device))
    if 'task3' in config['task_name']:
        with open(os.path.join(PICKLE_DIR, 'task3-TSR-dataset.pickle'), 'rb') as h:
            whole_dataset_dicts.append(torch.load(h, map_location=device))
    dataset = ZuCo_dataset(whole_dataset_dicts, 'test', None, subject=config['subjects'], eeg_type=config['eeg_type'], bands=config['eeg_bands'], setting='unique_sent', test_input='EEG')
    eeg_samples = []
    for i in range(min(32, len(dataset))):
        eeg, *_ = dataset[i]
        eeg_samples.append(eeg.unsqueeze(0))
    eeg_batch = torch.cat(eeg_samples, dim=0).to(device)  # (B, L, F)
    bands = config['eeg_bands']
    plot_band_attention(model, eeg_batch, bands)
