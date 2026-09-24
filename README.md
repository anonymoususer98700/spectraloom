# SpectraLoom

Sentence-level EEG-to-text generation with ZuCo 1.0 training and ZuCo 2.0 zero-shot evaluation.

SpectraLoom uses word-aligned gaze-duration EEG features (eight frequency bands, 105 electrodes per band), spectral band attention, parallel temporal convolutions with kernels **3/5/7**, a packed BiLSTM, an EEG-conditioned attention bridge, and BART-large. It generates sentences with **five-beam search**. This is not continuous raw-EEG thought decoding; sentence-level output does not imply sentence-averaged input.

## Results at a glance

Free-generation scores below are mean ± sample SD over the fixed seeds **42, 100, 312**. BLEU and ROUGE-L are on a 0–100 scale. The ZuCo 1.0 test set has 1,127 trials; ZuCo 2.0 zero-shot has 6,629.

| System | ZuCo 1.0 BLEU-1 | BLEU-4 | ROUGE-L | WER ↓ | ZuCo 2.0 BLEU-1 | BLEU-4 |
|:--|--:|--:|--:|--:|--:|--:|
| SpectraLoom | 17.76 ± 0.74 | 0.38 ± 0.11 | 16.80 ± 0.99 | 97.55 ± 1.70 | 16.33 ± 0.60 | 0.69 ± 0.27 |
| BrainTranslator/BART | 15.63 ± 3.35 | 0.43 ± 0.31 | 13.42 ± 2.54 | 119.16 ± 7.61 | 18.32 ± 2.25 | 1.28 ± 0.57 |
| T5-large | 18.42 ± 0.32 | 0.98 ± 0.23 | 16.42 ± 0.28 | 101.20 ± 0.73 | 16.43 ± 0.86 | 0.77 ± 0.11 |
| Pegasus-XSum | 6.38 ± 3.33 | 0.03 ± 0.04 | 9.69 ± 3.04 | 97.95 ± 0.21 | 4.90 ± 3.00 | 0.02 ± 0.04 |
| EEG2Text-FA | 14.75 ± 3.85 | 0.14 ± 0.13 | 14.64 ± 1.78 | 102.00 ± 2.99 | 14.19 ± 5.15 | 0.34 ± 0.51 |

**EEG2Text-FA is a feature-input adaptation, not a reproduction of published raw-EEG EEG2Text.** It uses this repository's word-aligned features and the common BART backbone/protocol instead of raw EEG and EEG pretraining. The systems are compared on the same split and decoding policy, but are not parameter matched.

The four SpectraLoom ablations—uniform SBA, no multi-scale, no CAB, and static SBA—have all three seeds. Full SpectraLoom has higher BLEU-1 through BLEU-4 than every ablation on both datasets. The six paired EEG-reliance conditions are also complete for three seeds. Their results are mixed: Gaussian EEG and removing EEG affect scores, but mismatched and shuffled EEG have nearly unchanged aggregate BLEU. **Do not interpret these results as strong proof of sentence-specific EEG decoding.** Teacher-forced scores use gold preceding tokens and are not free-generation scores.

Complete per-seed predictions, metrics, and aggregates are in [`experiments/outputs/`](experiments/outputs/). Start with [`all_results.csv`](experiments/outputs/tables/all_results.csv), [`teacher_forced_summary.json`](experiments/outputs/teacher_forced_summary.json), and [`reliance_summary.json`](experiments/outputs/reliance_summary.json).

## Run the complete experiment suite

Run these **two commands in order** from the repository root in PowerShell. The first covers SpectraLoom, three-seed component ablations, the historical seed-312 trained information controls, teacher forcing for SpectraLoom, and all paired reliance tests. The second covers the four three-seed baselines, their free-generation evaluations, and teacher-forced test diagnostics. Both regenerate aggregate tables and resume/skip completed work.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_spectraloom.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_baselines.ps1
```

Check what would run without training or modifying results:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_spectraloom.ps1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_baselines.ps1 -PlanOnly
```

**For an independent fresh reproduction**, use a new output directory. A clone contains result files and completion markers but not model checkpoints, so running against the published `experiments/outputs/` directory cannot recreate missing weights. The following commands leave the published results untouched and write to an ignored local directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_spectraloom.ps1 -OutputRoot .\experiments\reproduction_outputs
powershell -ExecutionPolicy Bypass -File .\scripts\run_baselines.ps1 -OutputRoot .\experiments\reproduction_outputs
```

Training resumes at the next epoch boundary from a temporary `resume.pt` where available. `complete.json` marks completed training; `best.pt` is needed to regenerate missing predictions. Existing evaluation files are checked against the saved five-beam decoding policy. Reliance evaluations additionally check their saved intervention signatures. A changed implementation, dataset, or training configuration requires a **new output directory**—do not reuse old completion markers for a different experiment. Neither script intentionally deletes model checkpoints.

## Environment and data

The recorded run used Windows 11, Python 3.12.10, PyTorch 2.10.0/CUDA 12.8, Transformers 5.3.0, and an NVIDIA RTX 5080. See [`provenance.json`](experiments/outputs/provenance.json) and each run's `config.json` for exact versions, seeds, hashes, hyperparameters, and timing. Reproduction across different GPUs or library versions is not guaranteed to be bitwise identical.

Create the local environment (or use an existing compatible one):

```powershell
py -3.12 -m venv .\EEG-To-text\venv
& .\EEG-To-text\venv\Scripts\python.exe -m pip install --upgrade pip
& .\EEG-To-text\venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
& .\EEG-To-text\venv\Scripts\python.exe -m pip install -r .\EEG-To-text\requirements.txt
& .\EEG-To-text\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Place the four prepared files in `EEG-To-text/Data/pickle_file/`:

| File | Used for |
|:--|:--|
| `task1-SR-dataset.pickle` | ZuCo 1.0 sentiment reading |
| `task2-NR-dataset.pickle` | ZuCo 1.0 natural reading |
| `task3-TSR-dataset.pickle` | ZuCo 1.0 task-specific reading |
| `task2-TSR-2.0-dataset.pickle` | ZuCo 2.0 zero-shot task-specific reading |

The conversion scripts for raw MATLAB data are in `EEG-To-text/Mat to Pickle file/`. Do not reconvert existing final pickles merely to resume a run. Only open trusted pickle/checkpoint files, and respect the datasets' distribution terms. Pretrained `facebook/bart-large`, `t5-large`, and `google/pegasus-xsum` weights/tokenizers are downloaded separately.

## Protocol

| Item | Recorded setting |
|:--|:--|
| Split | Sentence-grouped ZuCo 1.0 source; ZuCo 2.0 TSR held out entirely; split seed 2026 |
| Valid trials | 8,978 train / 1,105 dev / 1,127 test / 6,629 zero-shot |
| Training seeds | 42, 100, 312 for the main systems and four component ablations |
| Historical trained information controls | Length-only, language-prior, noise-only; seed 312 only |
| Normalization | Training-set feature statistics only |
| Optimization | AdamW, EEG/custom LR 2e-5, pretrained LR 2e-6, weight decay 0.01 |
| Batch and precision | Physical batch 8, gradient accumulation 8, BF16 |
| Selection | Up to 30 epochs; patience 5 on development loss |
| Free generation | 5 beams, max length 32, length penalty 1.4, repetition penalty 1.5, no-repeat 2-grams |
| Teacher forcing | Gold-prefix next-token argmax; reported separately |

Training histories and `complete.json` record per-epoch and total time. Logs are saved locally under `experiments/outputs/logs/` and ignored by Git. The paired reliance subset contains 782 eligible test trials and 4,659 eligible zero-shot trials; compare each condition with its **paired real-EEG row**, not with the full-split main table. Wrong-subject EEG keeps sentence identity, so unchanged scores are not inherently a failed control.

## Repository map

```text
.
├── README.md
├── scripts/
│   ├── run_spectraloom.ps1       # SpectraLoom, ablations, controls, reliance
│   └── run_baselines.ps1         # BrainTranslator, T5, Pegasus, EEG2Text-FA
├── EEG-To-text/
│   ├── model_decoding.py         # SpectraLoom and common decoder modules
│   ├── model_static_sba.py       # Static-band-weight control
│   ├── model_feature_eeg2text.py # EEG2Text feature-input adaptation
│   ├── Data.py                   # Feature/sample preparation
│   └── requirements.txt
├── experiments/
│   ├── train_experiment.py       # Deterministic training and checkpoints
│   ├── evaluate_experiment.py    # Free generation and paired EEG controls
│   ├── evaluate_teacher_forced.py
│   ├── aggregate_*.py           # Seed summaries
│   ├── manifests/               # Fixed leakage-aware split
│   ├── src/                     # Protocol, metrics, data, and I/O modules
│   ├── tests/                   # Implementation and protocol tests
│   └── outputs/                 # Published result files and local checkpoints
└── archive/                     # Superseded local artifacts; Git-ignored
```

The repository tracks code, split manifests, configurations, histories, predictions, normalization statistics, and summary tables. Raw datasets, virtual environments, runtime logs, archived experiments, and `*.pt` checkpoints remain local and are ignored by Git. A fresh clone needs its own data and model downloads for full reproduction.

Run the checks with:

```powershell
& .\EEG-To-text\venv\Scripts\python.exe -m pytest -q .\experiments\tests
```
