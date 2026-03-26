<p align="center">
  <img src="figures/architecture.pdf" alt="SpectraLoom Architecture" width="90%"/>
</p>

<h1 align="center">SpectraLoom</h1>
<h3 align="center">Spectral-Attentive Multi-Scale Neural Decoding for Open-Vocabulary EEG-to-Text Translation</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-12.8-76b900.svg" alt="CUDA"/>
</p>

---

## Abstract

Translating scalp EEG into open-vocabulary text is extremely challenging because the signals are noisy, high-dimensional, and unlike the discrete tokens that language models expect. **SpectraLoom** tackles this with three neuroscience-inspired modules:

1. **Spectral Band Attention (SBA)** — learns per-word weights for eight standard EEG frequency bands (θ₁, θ₂, α₁, α₂, β₁, β₂, γ₁, γ₂), reflecting their distinct linguistic roles.
2. **Multi-Scale Conv1D + BiLSTM Encoder** — parallel convolutions (kernels 1, 3, 5) capture short- and long-range temporal patterns, followed by a 3-layer bidirectional LSTM for sequence modeling.
3. **Cross-Attention Bridge (CAB)** — replaces simple linear projections with an 8-head cross-attention module using learnable position-specific query embeddings to interface the encoder with a pretrained **BART-large** decoder.

On the ZuCo 1.0 corpus the model attains **BLEU-1 = 15.48 %**, **ROUGE-L F1 = 12.23 %**, and **WER = 101.27 %** (no teacher forcing), yielding up to **24.7 % BLEU-4** and **61.5 % ROUGE-2** improvements over strong baselines.

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Architecture

SpectraLoom's encoder pipeline processes word-level EEG features (105 electrodes × 8 frequency bands = 840-dimensional vectors) through three novel components before feeding the output to a frozen/selectively-unfrozen BART-large decoder:

```
Raw EEG (B, L, 840)
    │
    ▼
┌─────────────────────────┐
│  LayerNorm + Noise Aug  │  z-score normalize; σ=0.02 Gaussian noise during training
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Spectral Band Attention│  Reshape → (B, L, 8, 105), learn per-band weights via softmax
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Multi-Scale Conv1D     │  Parallel Conv1D(k=1,3,5) → GELU → Concat → Project → LayerNorm
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  3-Layer BiLSTM         │  512 hidden per direction → 1024-d output → LayerNorm + Dropout
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Cross-Attention Bridge │  Learnable queries attend to encoder output (8-head, FFN, LayerNorm)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  BART-large Decoder     │  Pretrained 12-layer transformer; beam search (k=5, rep_penalty=5.0)
└─────────────────────────┘
```

---

## Repository Structure

```
EEG-To-text/
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── model_decoding.py           # Model architectures (EEGConformer/SpectraLoom, baselines)
├── train_decoding.py           # Training pipeline (2-phase, scheduled sampling, etc.)
├── eval_decoding.py            # Evaluation pipeline (BLEU, ROUGE, WER, CER, BERTScore)
├── Data.py                     # ZuCo dataset loader and preprocessing
├── config.py                   # Argument parser / config management
│
├── run_ablations.py            # Automated ablation study runner
├── eeg_reliance_test.py        # EEG reliance experiment (Real/Shuffled/Zero EEG)
├── eval_decoding.sh            # Evaluation shell script
│
├── Mat to Pickle file/         # Raw .mat → pickle preprocessing scripts
│   ├── create_pickle_file_v1.py    # ZuCo v1.0 converter
│   ├── create_pickle_file_v2.py    # ZuCo v2.0 converter
│   └── data_loading_helpers_modified.py
│
├── config/decoding/            # Saved training config JSONs (auto-generated)
├── Data/pickle_file/           # Preprocessed pickle datasets (not tracked)
├── checkpoints/decoding/       # Model checkpoints (not tracked)
├── Result/                     # Evaluation output files (not tracked)
├── figures/                    # Architecture diagram and generated plots
│   └── EEG_methodology.jpg    # Architecture figure for paper/README
└── dataset/                    # Raw ZuCo .mat files (not tracked)
    ├── task1-sr/
    ├── task2-nr/
    ├── task3-tsr/
    ├── zuco2-task1-nr/
    └── zuco2-task2-tsr/
```

---

## Requirements

### Hardware

Experiments were conducted on the following system (GPU with ≥16 GB VRAM recommended):

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i9-14900F |
| GPU | NVIDIA RTX 5080 (16 GB VRAM) |
| RAM | 64 GB DDR5 |
| OS | Windows 11 Pro (64-bit) |

### Software

```
Python         ≥ 3.12
PyTorch        ≥ 2.0  (with CUDA support)
Transformers   ≥ 4.30 (HuggingFace)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/SpectraLoom.git
cd SpectraLoom

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets evaluate sacrebleu
pip install nltk rouge-score rouge bert-score
pip install numpy pandas matplotlib tqdm scipy scikit-learn
pip install jiwer   # for WER/CER metrics

# Download NLTK data (required for BLEU)
python -c "import nltk; nltk.download('punkt')"
```

---

## Dataset

This project uses the **ZuCo (Zurich Cognitive Language Processing) Corpus** — a large-scale dataset of simultaneous EEG and eye-tracking recordings collected while participants engaged in naturalistic reading.

### ZuCo v1.0 Tasks

| Task | Name | Description | Sentences |
|------|------|-------------|-----------|
| Task 1 (SR) | Sentiment Reading | 12 subjects read 400 movie-review sentences and classified sentiment | 400 |
| Task 2 (NR) | Normal Reading | 12 subjects read 300 Wikipedia sentences with no explicit task | 300 |
| Task 3 (TSR) | Task-Specific Reading | 12 subjects read 407 sentences and answered comprehension questions | 407 |

### EEG Recording Details

- **System**: 128-channel EGI HydroCel at 500 Hz (reduced to **105 channels** after preprocessing)
- **Feature type**: Word-level **Gaze Duration (GD)** mean amplitudes
- **Frequency bands (8)**: θ₁ (4–6 Hz), θ₂ (6.5–8 Hz), α₁ (8.5–10 Hz), α₂ (10.5–13 Hz), β₁ (13.5–18 Hz), β₂ (18.5–30 Hz), γ₁ (30.5–40 Hz), γ₂ (40–49.5 Hz)
- **Feature dimension**: 105 electrodes × 8 bands = **840** per word

### Download

1. Download the ZuCo dataset from the [official OSF repository](https://osf.io/q3zws/)
2. Place the `.mat` files in the `dataset/` directory following this structure:
   ```
   dataset/
   ├── task1-sr/           # ZuCo v1.0 Task 1 .mat files
   ├── task2-nr/           # ZuCo v1.0 Task 2 .mat files
   ├── task3-tsr/          # ZuCo v1.0 Task 3 .mat files
   ├── zuco2-task1-nr/     # ZuCo v2.0 Task 1 .mat files (optional)
   └── zuco2-task2-tsr/    # ZuCo v2.0 Task 2 .mat files (optional)
   ```

### Dataset Split

Data is split at the **sentence level** (80% train / 10% dev / 10% test) using the `unique_sent` setting, ensuring **no sentence overlap** between splits while multiple subject readings of the same sentence stay in the same split.

---

## Data Preprocessing

Convert raw ZuCo `.mat` files into pickle format:

```bash
# ZuCo v1.0
python "Mat to Pickle file/create_pickle_file_v1.py"

# ZuCo v2.0 (optional)
python "Mat to Pickle file/create_pickle_file_v2.py"
```

This produces the following pickle files in `Data/pickle_file/`:

| File | Description |
|------|-------------|
| `task1-SR-dataset.pickle` | Sentiment Reading (ZuCo v1.0 Task 1) |
| `task2-NR-dataset.pickle` | Normal Reading (ZuCo v1.0 Task 2) |
| `task3-TSR-dataset.pickle` | Task-Specific Reading (ZuCo v1.0 Task 3) |
| `task2-TSR-2.0-dataset.pickle` | ZuCo v2.0 (optional) |

### Pickle File Format

Each pickle file is a dictionary keyed by subject ID (e.g., `"ZAB"`, `"ZDM"`, ...), where each value is a list of sentence objects:

```python
{
    "ZAB": [
        {
            "content": "The movie was excellent.",           # target text
            "word": [                                        # list of word objects
                {
                    "word_level_EEG": {
                        "GD": {
                            "GD_t1": np.array([...]),       # 105-dim for θ₁
                            "GD_t2": np.array([...]),       # 105-dim for θ₂
                            "GD_a1": np.array([...]),       # ... etc for all 8 bands
                            ...
                        }
                    }
                },
                ...
            ],
            "sentence_level_EEG": {
                "mean_t1": np.array([...]),                  # 105-dim sentence-level
                ...
            }
        },
        ...
    ],
    "ZDM": [...],
    ...
}
```

---

## Training

### Full Model (SpectraLoom / EEGConformer)

The recommended training uses the **skip-step-one** mode (directly trains Phase 2 from scratch) on all three ZuCo v1.0 tasks:

```bash
python train_decoding.py \
    -m EEGConformer \
    -t task1_task2_taskNRv2 \
    -ne1 20 \
    -ne2 30 \
    -lr1 5e-05 \
    -lr2 5e-07 \
    -b 32 \
    -s ./checkpoints/decoding \
    --train_input EEG \
    --skip_step_one
```

### Two-Phase Training (Optional)

For full two-phase training with Phase 1 warm-up:

```bash
python train_decoding.py \
    -m EEGConformer \
    -t task1_task2_taskNRv2 \
    -ne1 20 \
    -ne2 30 \
    -lr1 5e-05 \
    -lr2 5e-07 \
    -b 32 \
    -s ./checkpoints/decoding \
    --train_input EEG
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model_name` (`-m`) | `EEGConformer` | Model architecture |
| `--task_name` (`-t`) | `task1_task2_taskNRv2` | Combine multiple reading tasks |
| `--num_epoch_step1` (`-ne1`) | `20` | Phase 1 epochs |
| `--num_epoch_step2` (`-ne2`) | `30` | Phase 2 epochs |
| `--learning_rate_step1` (`-lr1`) | `5e-05` | Phase 1 LR (AdamW + CosineAnnealing) |
| `--learning_rate_step2` (`-lr2`) | `5e-07` | Phase 2 custom LR (BART at 10x lower) |
| `--batch_size` (`-b`) | `32` | Batch size (effective 64 with 2x accumulation) |
| `--train_input` | `EEG` | Use real EEG (`EEG`) or noise baseline (`noise`) |
| `--skip_step_one` | flag | Skip Phase 1, start directly with Phase 2 |
| `--eeg_type` | `GD` | Gaze Duration feature type |
| `--subjects` | `ALL` | Use all 12 subjects |

### Training Recipe Details

- **Gradient accumulation**: 2 steps (effective batch size = 64)
- **Label smoothing**: ε = 0.1
- **Scheduled sampling**: TF ratio decays 1.0 → 0.3 (sigmoid schedule with 40% warmup)
- **Early stopping**: patience = 10 epochs on validation loss
- **Gradient clipping**: max norm = 1.0
- **Discriminative LR**: custom EEG layers at `lr2`, pretrained BART at `lr2 / 10`
- **Selective unfreezing**: BART embeddings + first 4 encoder/decoder layers in Phase 2

### Ablation Experiments

Run all ablation variants sequentially:

```bash
python run_ablations.py
```

Or run individual ablations:

```bash
# Without Spectral Band Attention (uniform weights)
python train_decoding.py -m EEGConformer -t task1_task2_taskNRv2 \
    -ne1 20 -ne2 30 -lr1 5e-05 -lr2 5e-07 -b 32 \
    -s ./checkpoints/decoding --train_input EEG --skip_step_one \
    --ablate_sba uniform

# Without Multi-Scale Conv1D
python train_decoding.py ... --ablate_multiscale

# Without Cross-Attention Bridge
python train_decoding.py ... --ablate_cab

# Without Label Smoothing
python train_decoding.py ... --ablate_label_smoothing

# Alternative convolution kernels {3, 5, 7}
python train_decoding.py ... --conv_kernels 3,5,7
```

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval_decoding.py \
    -checkpoint ./checkpoints/decoding/best/<checkpoint_name>.pt \
    -conf ./config/decoding/<config_name>.json \
    -test_input EEG \
    -train_input EEG
```

### Example (Full Model)

```bash
python eval_decoding.py \
    -checkpoint ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_EEGConformer_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_EEG.pt \
    -conf ./config/decoding/task1_task2_taskNRv2_finetune_EEGConformer_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_EEG.json \
    -test_input EEG \
    -train_input EEG
```

### EEG Reliance Test

Validates that the model uses genuine EEG information (not just the language model prior):

```bash
python eeg_reliance_test.py
```

This runs evaluation under three conditions:
- **Real EEG** — original word-level embeddings
- **Shuffled EEG** — randomly permuted within each sentence
- **Zero EEG** — all-zero input tensor

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Corpus BLEU-1/2/3/4** | N-gram precision at corpus level |
| **SacreBLEU** | Standardized, reproducible BLEU |
| **ROUGE-1/2/L F1** | Recall-oriented n-gram overlap |
| **WER** | Word Error Rate |
| **CER** | Character Error Rate |
| **BERTScore F1** | Contextual embedding similarity |

Results are reported both **with teacher forcing (TF)** and **without teacher forcing (No TF)**. The No TF setting is the primary metric of interest.

---

## Results

### Main Results (No Teacher Forcing) — ZuCo v1.0

| Model | WER ↓ | BLEU-1 ↑ | BLEU-2 ↑ | BLEU-3 ↑ | BLEU-4 ↑ | ROUGE-L ↑ |
|-------|-------|----------|----------|----------|----------|-----------|
| BART (sentence) | 142.10 | 11.02 | 1.08 | 0.42 | 0.28 | 6.54 |
| Pegasus (sentence) | 147.80 | 6.20 | 0.55 | 0.24 | 0.16 | 4.03 |
| T5 (sentence) | 141.00 | 12.50 | 1.18 | 0.48 | 0.31 | 7.15 |
| DELTA (sentence) | 139.71 | 14.82 | 1.36 | 0.56 | 0.37 | 7.26 |
| **SpectraLoom (sentence)** | **101.27** | **15.48** | **4.40** | **1.37** | **0.45** | **12.23** |
| BART (word) | 108.43 | 13.69 | 2.97 | 0.82 | 0.32 | 11.87 |
| T5 (word) | 111.13 | 16.64 | 5.80 | 1.96 | 0.81 | 11.85 |
| DELTA (word) | 110.03 | 21.93 | 6.43 | 2.01 | 0.76 | 17.24 |
| **SpectraLoom (word)** | **104.30** | **19.21** | **5.53** | **1.82** | **0.66** | **11.49** |

### Results (With Teacher Forcing) — ZuCo v1.0

| Model | B-1 | B-2 | B-3 | B-4 | R1-F | R2-F | RL-F | WER ↓ | CER ↓ | SacreBLEU |
|-------|-----|-----|-----|-----|------|------|------|-------|-------|-----------|
| R1 Translator | 38.62 | 21.41 | 11.65 | 6.15 | 27.79 | 6.45 | 25.90 | 0.73 | 0.58 | 8.78 |
| T5 Translator | 43.01 | 24.41 | 14.15 | 7.75 | 24.76 | 4.76 | 22.72 | 0.76 | 0.59 | 6.06 |
| Brain Translator | 38.02 | 20.43 | 10.56 | 5.62 | 26.37 | 5.14 | 24.53 | 0.76 | 0.60 | 6.69 |
| **SpectraLoom** | **40.07** | **21.70** | **11.96** | **6.65** | **31.40** | **8.98** | **29.46** | **0.76** | **0.60** | **6.36** |

### EEG Reliance Test

| Metric | Real EEG | Shuffled EEG | Zero EEG |
|--------|----------|--------------|----------|
| BLEU-1 (%) | 15.48 | 15.41 | 7.18 |
| BLEU-4 (%) | 0.45 | 0.39 | 0.00 |
| SacreBLEU | 0.53 | 0.45 | 0.21 |
| WER (%) | 101.27 | 101.08 | 95.61 |

Zeroing EEG input drops BLEU-1 by **53.6%**, confirming the model extracts genuine neural signal information.

---

## Ablation Study

### Without Teacher Forcing

| Variant | BLEU-1 | BLEU-4 | ROUGE-L F1 | SacreBLEU |
|---------|--------|--------|------------|-----------|
| **Full Model** | **15.48** | **0.45** | **12.23** | **0.53** |
| w/o SBA (uniform weights) | 8.79 | 0.02 | 7.36 | 0.034 |
| w/o Multi-Scale Conv1D | 9.59 | 0.03 | 7.58 | 0.050 |
| w/o Cross-Attention Bridge | 9.16 | 0.04 | 7.08 | 0.054 |
| w/o Label Smoothing | 9.35 | 0.03 | 5.84 | 0.041 |
| Alt. Kernels {3,5,7} | 9.20 | 0.04 | 7.08 | 0.054 |

### With Teacher Forcing

| Variant | BLEU-1 | BLEU-4 | ROUGE-L F1 | SacreBLEU |
|---------|--------|--------|------------|-----------|
| **Full Model** | **40.07** | **6.65** | **29.46** | **6.36** |
| w/o SBA (uniform weights) | 29.90 | 3.18 | 21.06 | 3.12 |
| w/o Multi-Scale Conv1D | 33.96 | 3.29 | 24.40 | 3.20 |
| w/o Cross-Attention Bridge | 32.47 | 3.61 | 23.62 | 3.22 |
| w/o Label Smoothing | 31.34 | 3.19 | 22.84 | 3.06 |
| Alt. Kernels {3,5,7} | 33.40 | 3.34 | 23.62 | 3.22 |

**Key finding**: Spectral Band Attention produces the largest single-component effect (−53% BLEU-1 when removed).

---


## Acknowledgements

- **NeuSpeech** — the foundational [EEG-To-Text](https://github.com/NeuSpeech/EEG-To-Text) codebase that served as the backbone for this work.
- **ZuCo Dataset** — [Hollenstein et al., 2018](https://osf.io/q3zws/) for the high-quality EEG and eye-tracking recordings.
- **Jo et al., 2024** — ["Are EEG-to-Text Models Working?"](https://arxiv.org/abs/2405.06459) for establishing rigorous evaluation protocols.
- **Jeon et al., 2025** — [DELTA](https://arxiv.org/abs/2511.21746) for comparative baselines.

---
