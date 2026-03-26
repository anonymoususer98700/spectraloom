"""
EEG Reliance Test
=================
Runs the EEG-to-Text model three ways on the test set:
  1. Real EEG       – normal EEG embeddings (as-trained)
  2. Shuffled EEG   – temporal/sequence order of words shuffled within each sample
  3. Zero EEG       – all-zeros tensor (no signal at all)

Saves a unified results file:  Result/eeg_reliance_test_scores.txt
Also saves per-condition prediction CSVs.

Usage (from EEG-To-text directory, venv activated):
    python eeg_reliance_test.py \
        -checkpoint ./checkpoints/decoding/best/task1v1_task2v1_task3v1_finetune_EEGConformer_skipstep1_b32_20_30_0.0001_2e-05_unique_sent_EEG.pt \
        -conf ./config/decoding/task1v1_task2v1_task3v1_finetune_EEGConformer_skipstep1_b32_20_30_0.0001_2e-05_unique_sent_EEG.json
"""

import os
import sys
import json
import argparse
import time
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

# Local imports
from Data import ZuCo_dataset
from model_decoding import EEGConformer
import evaluate as hf_evaluate

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
sacrebleu_metric = hf_evaluate.load("sacrebleu")
cer_metric = hf_evaluate.load("cer")
wer_metric = hf_evaluate.load("wer")

WEIGHTS_LIST = [
    (1.0,),
    (0.5, 0.5),
    (1.0 / 3, 1.0 / 3, 1.0 / 3),
    (0.25, 0.25, 0.25, 0.25),
]


def remove_eos(text, token="</s>"):
    idx = text.find(token)
    return text[:idx] if idx != -1 else text


# ---------------------------------------------------------------------------
# EEG-input transforms
# ---------------------------------------------------------------------------

def make_shuffled_loader(real_loader):
    """Return a list of batches where the word-order dimension is shuffled."""
    shuffled_batches = []
    for batch in real_loader:
        emb, seq_len, masks, masks_inv, tgt_ids, tgt_mask, sent_labels = batch
        # emb shape: (B, max_len, feat)
        B, L, F = emb.shape
        shuffled_emb = emb.clone()
        for b in range(B):
            sl = int(seq_len[b].item())
            if sl > 1:
                perm = torch.randperm(sl)
                shuffled_emb[b, :sl] = emb[b, perm]
        shuffled_batches.append((shuffled_emb, seq_len, masks, masks_inv, tgt_ids, tgt_mask, sent_labels))
    return shuffled_batches


def make_zero_loader(real_loader):
    """Return a list of batches where all EEG embeddings are zero."""
    zero_batches = []
    for batch in real_loader:
        emb, seq_len, masks, masks_inv, tgt_ids, tgt_mask, sent_labels = batch
        zero_batches.append((torch.zeros_like(emb), seq_len, masks, masks_inv, tgt_ids, tgt_mask, sent_labels))
    return zero_batches


# ---------------------------------------------------------------------------
# Evaluation for one condition
# ---------------------------------------------------------------------------

def eval_condition(batch_iter, device, tokenizer, model, condition_name, out_dir):
    """
    Run model inference over all batches in batch_iter and return metric scores.
    batch_iter can be a DataLoader or a list of pre-built batches.
    """
    model.eval()

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []

    smoothing = SmoothingFunction().method1
    txt_path = os.path.join(out_dir, f"eeg_reliance_{condition_name}_predictions.txt")
    csv_path = os.path.join(out_dir, f"eeg_reliance_{condition_name}_predictions.csv")

    with torch.no_grad():
        with open(txt_path, "w", encoding="utf-8") as f:
            for batch in tqdm(batch_iter, desc=f"[{condition_name}]"):
                emb, seq_len, masks, masks_inv, tgt_ids, tgt_mask, _ = batch

                emb = emb.to(device).float()
                masks = masks.to(device)
                masks_inv = masks_inv.to(device)
                tgt_ids = tgt_ids.to(device)

                # --- target ---
                tgt_string = tokenizer.decode(tgt_ids[0], skip_special_tokens=True)
                tgt_tokens = tokenizer.convert_ids_to_tokens(
                    tgt_ids[0].tolist(), skip_special_tokens=True
                )
                target_string_list.append(tgt_string)
                target_tokens_list.append([tgt_tokens])

                # Replace pad with -100 for the loss (needed by model signature)
                tgt_ids_in = tgt_ids.clone()
                tgt_ids_in[tgt_ids_in == tokenizer.pad_token_id] = -100

                f.write(f"target: {tgt_string}\n")

                # --- generate ---
                preds = model.generate(
                    emb,
                    masks,
                    masks_inv,
                    tgt_ids_in,
                    max_length=56,
                    num_beams=5,
                    do_sample=False,
                    repetition_penalty=2.5,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0,
                    early_stopping=True,
                )
                pred_string = tokenizer.batch_decode(preds, skip_special_tokens=True)[0]
                f.write(f"predicted: {pred_string}\n")
                f.write("=" * 60 + "\n")

                pred_ids = tokenizer.encode(pred_string)
                truncated = []
                for t in pred_ids:
                    if t != tokenizer.eos_token_id:
                        truncated.append(t)
                    else:
                        break
                pred_toks = tokenizer.convert_ids_to_tokens(truncated, skip_special_tokens=True)
                pred_tokens_list.append(pred_toks)
                pred_string_list.append(pred_string)

    # Save CSV
    df = pd.DataFrame({"target": target_string_list, "predicted": pred_string_list})
    df.to_csv(csv_path, index=False)
    print(f"  [{condition_name}] saved predictions -> {csv_path}")

    # ---------- compute metrics ----------
    bleu_scores = []
    for w in WEIGHTS_LIST:
        s = corpus_bleu(target_tokens_list, pred_tokens_list, weights=w)
        bleu_scores.append(round(s * 100, 4))

    ref_list = [[t] for t in target_string_list]
    sacre = sacrebleu_metric.compute(predictions=pred_string_list, references=ref_list)

    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True, ignore_empty=True)
    except ValueError:
        rouge_scores = "Hypothesis is empty (all preds empty)"

    wer_score = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    cer_score = cer_metric.compute(predictions=pred_string_list, references=target_string_list)

    # Per-sample BLEU-1
    per_sample_bleu1 = []
    for tgt, pred in zip(target_string_list, pred_string_list):
        t_toks = tgt.split()
        p_toks = pred.split()
        if len(p_toks) == 0:
            per_sample_bleu1.append(0.0)
        else:
            per_sample_bleu1.append(
                sentence_bleu([t_toks], p_toks, weights=(1.0,), smoothing_function=smoothing)
            )
    mean_sent_bleu1 = round(float(np.mean(per_sample_bleu1)) * 100, 4)

    metrics = {
        "condition": condition_name,
        "BLEU-1 (corpus, %)": bleu_scores[0],
        "BLEU-2 (corpus, %)": bleu_scores[1],
        "BLEU-3 (corpus, %)": bleu_scores[2],
        "BLEU-4 (corpus, %)": bleu_scores[3],
        "Mean Sentence BLEU-1 (%)": mean_sent_bleu1,
        "SacreBLEU score": round(sacre["score"], 4) if isinstance(sacre, dict) else sacre,
        "ROUGE scores": rouge_scores,
        "WER (%)": round(wer_score * 100, 4),
        "CER (%)": round(cer_score * 100, 4),
    }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EEG Reliance Test")
    parser.add_argument(
        "-checkpoint", "--checkpoint_path",
        required=True,
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "-conf", "--config_path",
        required=True,
        help="Path to training config JSON"
    )
    parser.add_argument(
        "-seed", "--seed",
        type=int,
        default=20,
        help="Random seed (default: 20)"
    )
    args = parser.parse_args()

    # ---- seeds ----
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- load training config ----
    training_config = json.load(open(args.config_path))
    subject_choice = training_config["subjects"]
    eeg_type_choice = training_config["eeg_type"]
    bands_choice = training_config["eeg_bands"]
    task_name = training_config["task_name"]
    model_name = training_config["model_name"]

    print(f"[INFO] Task: {task_name}  |  Model: {model_name}")
    print(f"[INFO] EEG type: {eeg_type_choice}  |  Bands: {bands_choice}")
    print(f"[INFO] Subjects: {subject_choice}")

    # ---- tokenizer ----
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    # ---- load pickle datasets ----
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PICKLE_DIR = os.path.join(BASE_DIR, "Data", "pickle_file")
    whole_dataset_dicts = []
    if "task1" in task_name:
        with open(os.path.join(PICKLE_DIR, "task1-SR-dataset.pickle"), "rb") as h:
            whole_dataset_dicts.append(pickle.load(h))
    if "task2" in task_name:
        with open(os.path.join(PICKLE_DIR, "task2-NR-dataset.pickle"), "rb") as h:
            whole_dataset_dicts.append(pickle.load(h))
    if "task3" in task_name:
        with open(os.path.join(PICKLE_DIR, "task2-TSR-2.0-dataset.pickle"), "rb") as h:
            whole_dataset_dicts.append(pickle.load(h))

    dataset_setting = "unique_sent"

    # ---- test set with real EEG ----
    print("\n[INFO] Loading test set (real EEG) ...")
    test_set = ZuCo_dataset(
        whole_dataset_dicts, "test", tokenizer,
        subject=subject_choice,
        eeg_type=eeg_type_choice,
        bands=bands_choice,
        setting=dataset_setting,
        test_input="EEG",      # real EEG
    )
    print(f"[INFO] Test set size: {len(test_set)}")

    real_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # ---- build shuffled and zero batches from real loader ----
    print("[INFO] Pre-fetching real batches to build shuffled/zero variants ...")
    real_batches = list(real_loader)          # list of tuples
    shuffled_batches = make_shuffled_loader(real_batches)
    zero_batches = make_zero_loader(real_batches)
    print(f"[INFO] Total test samples: {len(real_batches)}")

    # ---- load model ----
    print("\n[INFO] Loading model ...")
    ablate_sba = training_config.get("ablate_sba", False)
    ablate_multiscale = training_config.get("ablate_multiscale", False)
    ablate_cab = training_config.get("ablate_cab", False)
    ablate_label_smoothing = training_config.get("ablate_label_smoothing", False)
    
    conv_kernels_str = training_config.get("conv_kernels", "1,3,5")
    if isinstance(conv_kernels_str, str):
        conv_kernels = tuple(map(int, conv_kernels_str.split(",")))
    else:
        conv_kernels = (1, 3, 5)

    pretrained = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", torch_dtype=torch.float32
    )
    pretrained = pretrained.float()
    pretrained.config.label_smoothing = 0.0 if ablate_label_smoothing else 0.1

    model = EEGConformer(
        pretrained,
        in_feature=105 * len(bands_choice),
        decoder_embedding_size=pretrained.config.d_model,
        n_bands=len(bands_choice),
        n_electrodes=105,
        conv_channels=105 * len(bands_choice),
        rnn_hidden_size=512,
        num_rnn_layers=3,
        n_bridge_heads=8,
        dropout=0.2,
        ablate_sba=ablate_sba,
        ablate_multiscale=ablate_multiscale,
        ablate_cab=ablate_cab,
        conv_kernels=conv_kernels
    )
    state_dict = torch.load(args.checkpoint_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if "multi_scale_conv.conv1." in k: k = k.replace("conv1.", "convs.0.")
        elif "multi_scale_conv.conv3." in k: k = k.replace("conv3.", "convs.1.")
        elif "multi_scale_conv.conv5." in k: k = k.replace("conv5.", "convs.2.")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # ---- output directory ----
    RESULT_DIR = os.path.join(BASE_DIR, "Result")
    os.makedirs(RESULT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    SCORE_FILE = os.path.join(RESULT_DIR, f"eeg_reliance_test_scores_{timestamp}.txt")

    # ---- run three conditions ----
    all_results = []

    for condition_tag, batches in [
        ("Real_EEG", real_batches),
        ("Shuffled_EEG", shuffled_batches),
        ("Zero_EEG", zero_batches),
    ]:
        print(f"\n{'='*60}")
        print(f"  Running condition: {condition_tag}")
        print(f"{'='*60}")
        t0 = time.time()
        metrics = eval_condition(batches, device, tokenizer, model, condition_tag, RESULT_DIR)
        elapsed = time.time() - t0
        metrics["elapsed_seconds"] = round(elapsed, 1)
        all_results.append(metrics)
        print(f"  [{condition_tag}] Done in {elapsed/60:.1f} min")

    # ---- print comparison table ----
    print("\n" + "=" * 70)
    print("  EEG RELIANCE TEST – SUMMARY")
    print("=" * 70)
    metric_keys = [
        "BLEU-1 (corpus, %)",
        "BLEU-2 (corpus, %)",
        "BLEU-3 (corpus, %)",
        "BLEU-4 (corpus, %)",
        "Mean Sentence BLEU-1 (%)",
        "SacreBLEU score",
        "WER (%)",
        "CER (%)",
    ]
    header = f"{'Metric':<30}" + "".join(f"{r['condition']:>18}" for r in all_results)
    print(header)
    print("-" * len(header))
    for k in metric_keys:
        row = f"{k:<30}" + "".join(f"{r[k]:>18}" for r in all_results)
        print(row)
    print("=" * 70)

    # ---- save full results to text file ----
    with open(SCORE_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  EEG RELIANCE TEST RESULTS\n")
        f.write(f"  Checkpoint : {args.checkpoint_path}\n")
        f.write(f"  Config     : {args.config_path}\n")
        f.write(f"  Timestamp  : {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        # Summary table
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for k in metric_keys:
            row = f"{k:<30}" + "".join(f"{r[k]:>18}" for r in all_results)
            f.write(row + "\n")
        f.write("=" * 70 + "\n\n")

        # Full detail per condition
        for r in all_results:
            f.write(f"\n--- {r['condition']} ---\n")
            for k, v in r.items():
                f.write(f"  {k}: {v}\n")

        # Interpretation notes
        f.write("\n\n--- INTERPRETATION ---\n")
        real_b1   = next(r["BLEU-1 (corpus, %)"] for r in all_results if r["condition"] == "Real_EEG")
        shuf_b1   = next(r["BLEU-1 (corpus, %)"] for r in all_results if r["condition"] == "Shuffled_EEG")
        zero_b1   = next(r["BLEU-1 (corpus, %)"] for r in all_results if r["condition"] == "Zero_EEG")
        delta_shuf = round(real_b1 - shuf_b1, 4)
        delta_zero = round(real_b1 - zero_b1, 4)
        f.write(f"  Real vs Shuffled EEG BLEU-1 drop : {delta_shuf:.4f} %\n")
        f.write(f"  Real vs Zero EEG    BLEU-1 drop : {delta_zero:.4f} %\n")
        if delta_zero < 1.0:
            f.write("  WARNING: Very small drop vs zero EEG – model may not be relying on EEG signal.\n")
        elif delta_zero > 5.0:
            f.write("  GOOD: Large drop vs zero EEG – model is significantly relying on EEG signal.\n")
        else:
            f.write("  MODERATE: Some reliance on EEG signal detected.\n")

    print(f"\n[INFO] Full scores saved to:\n  {SCORE_FILE}")
    print("[INFO] Per-condition prediction CSVs saved to the Result/ directory.")


if __name__ == "__main__":
    main()
