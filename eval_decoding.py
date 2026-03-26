import os
import csv
import argparse
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
import torch.nn.functional as F
import time
import re
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationDecoder
from Data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator, R1Translator, EEGConformer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from config import get_config
import evaluate
from evaluate import load

metric = evaluate.load("sacrebleu")
cer_metric = load("cer")
wer_metric = load("wer")
bertscore_metric = evaluate.load("bertscore")

def remove_text_after_token(text, token='</s>'):
    # 특정 토큰 이후의 텍스트를 찾아 제거
    token_index = text.find(token)
    if token_index != -1:  # 토큰이 발견된 경우
        return text[:token_index]  # 토큰 이전까지의 텍스트 반환
    return text  # 토큰이 없으면 원본 텍스트 반환

def tokenize_for_bleu(text: str):
    """
    Tokenizer for the NLTK corpus_bleu computation.
    Uses lowercased word/punctuation tokens (not BPE pieces) so BLEU-4
    is sensitive to real word order and stays numerically stable.
    """
    if text is None:
        return []
    text = text.strip().lower()
    if not text:
        return []
    # Words/numbers or single punctuation marks
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", text)

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = './results/temp.txt' , score_results='./score_results/task.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    start_time = time.time()
    model.eval()   # Set model to evaluate mode
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []


    with open(output_all_results_path,'w', encoding='utf-8') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels in tqdm(dataloaders['test']):
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float() # B, 56, 840
            input_masks_batch = input_masks.to(device) # B, 56
            target_ids_batch = target_ids.to(device) # B, 56
            input_mask_invert_batch = input_mask_invert.to(device) # B, 56
            target_mask_batch = target_mask.to(device) # B, 56
            
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_strininvert.to(device) # B, 56
            
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # target_ids_batch_label = target_ids_batch.clone().detach()
            # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100

            # Original code 
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch) # (batch, time, n_class)
            logits_previous = seq2seqLMoutput.logits  # (B, seq_len, vocab)
            pred_ids_previous = torch.argmax(logits_previous, dim=-1)[0]  # (seq_len,)

            # Avoid decoding padding positions by slicing with the reference attention mask.
            valid_len = int(target_mask_batch[0].sum().item())
            pred_ids_previous = pred_ids_previous[:valid_len]

            # Truncate at eos if present.
            eos_id = tokenizer.eos_token_id
            eos_positions = (pred_ids_previous == eos_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                pred_ids_previous = pred_ids_previous[:first_eos]

            predicted_string_previous = tokenizer.decode(pred_ids_previous.tolist(), skip_special_tokens=True)
            f.write(f'predicted string with tf: {predicted_string_previous}\n')

            pred_tokens_previous = tokenizer.convert_ids_to_tokens(pred_ids_previous.tolist(), skip_special_tokens=True)
            pred_tokens_list_previous.append(pred_tokens_previous)
            pred_string_list_previous.append(predicted_string_previous)
            

            predictions = model.generate(
   			 input_embeddings_batch, 
   			 input_masks_batch, 
   			 input_mask_invert_batch, 
  			 target_ids_batch,
  			 max_length=56,
   			 num_beams=5,
    			do_sample=False,
    			repetition_penalty=2.5,
    			no_repeat_ngram_size=2,
                        length_penalty=1.0,
                        early_stopping=True
			)            
            # Decode directly from generated ids (avoid manual eos truncation which can
            # lead to empty strings and BLEU-0 if eos appears early).
            predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # For logging/tokens list only: re-encode and remove eos tokens.
            pred_ids_for_tokens = tokenizer.encode(predicted_string)
            truncated_prediction = []
            for t in pred_ids_for_tokens:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # pred_tokens_list.extend(pred_tokens)
            # pred_string_list.extend(predicted_string)
            # print('################################################')
            # print()
    # print(f"pred_string_list : {pred_string_list}")

    #New Code
    csv_path = output_all_results_path.replace('.txt', '_predictions.csv')
    predictions_df = pd.DataFrame({
        'target_text':target_string_list,
        'predicted_text_no_tf': pred_string_list,
        'predicted_text_with_tf': pred_string_list_previous
    })
    predictions_df.to_csv(csv_path, index=False)
    print(f'Saved predictions to {csv_path}')

    # Ranked results by per-sample similarity
    from nltk.translate.bleu_score import SmoothingFunction
    from difflib import SequenceMatcher
    smoothing = SmoothingFunction().method1

    # --- File 1: Ranked by accuracy WITHOUT teacher forcing ---
    no_tf_scores = []
    for i, (tgt, pred) in enumerate(zip(target_string_list, pred_string_list)):
        tgt_tokens = tgt.split()
        pred_tokens_i = pred.split()
        if len(pred_tokens_i) == 0:
            bleu1 = 0.0
        else:
            bleu1 = sentence_bleu([tgt_tokens], pred_tokens_i, weights=(1.0,), smoothing_function=smoothing)
        seq_sim = SequenceMatcher(None, tgt.lower(), pred.lower()).ratio()
        no_tf_scores.append({
            'rank': 0,
            'sample_index': i,
            'bleu1_score': round(bleu1, 4),
            'string_similarity': round(seq_sim, 4),
            'target_text': tgt,
            'predicted_text': pred
        })
    no_tf_scores.sort(key=lambda x: (x['bleu1_score'], x['string_similarity']), reverse=True)
    for rank, item in enumerate(no_tf_scores, 1):
        item['rank'] = rank
    ranked_no_tf_df = pd.DataFrame(no_tf_scores)
    ranked_no_tf_path = output_all_results_path.replace('.txt', '_ranked_without_tf.csv')
    ranked_no_tf_df.to_csv(ranked_no_tf_path, index=False)
    print(f'Saved ranked predictions (WITHOUT TF) to {ranked_no_tf_path}')
    print(f'  [No TF] Top 5 BLEU-1: {[s["bleu1_score"] for s in no_tf_scores[:5]]}')

    # --- File 2: Ranked by accuracy WITH teacher forcing ---
    tf_scores = []
    for i, (tgt, pred_tf) in enumerate(zip(target_string_list, pred_string_list_previous)):
        tgt_tokens = tgt.split()
        pred_tokens_i = pred_tf.split()
        if len(pred_tokens_i) == 0:
            bleu1 = 0.0
        else:
            bleu1 = sentence_bleu([tgt_tokens], pred_tokens_i, weights=(1.0,), smoothing_function=smoothing)
        seq_sim = SequenceMatcher(None, tgt.lower(), pred_tf.lower()).ratio()
        tf_scores.append({
            'rank': 0,
            'sample_index': i,
            'bleu1_score': round(bleu1, 4),
            'string_similarity': round(seq_sim, 4),
            'target_text': tgt,
            'predicted_text': pred_tf
        })
    tf_scores.sort(key=lambda x: (x['bleu1_score'], x['string_similarity']), reverse=True)
    for rank, item in enumerate(tf_scores, 1):
        item['rank'] = rank
    ranked_tf_df = pd.DataFrame(tf_scores)
    ranked_tf_path = output_all_results_path.replace('.txt', '_ranked_with_tf.csv')
    ranked_tf_df.to_csv(ranked_tf_path, index=False)
    print(f'Saved ranked predictions (WITH TF) to {ranked_tf_path}')
    print(f'  [With TF] Top 5 BLEU-1: {[s["bleu1_score"] for s in tf_scores[:5]]}')
    #New Code End
    
    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    corpus_bleu_scores = []
    corpus_bleu_scores_previous = []

    # Compute corpus BLEU on word/punctuation tokens (not BPE pieces).
    references_word_tokens = [[tokenize_for_bleu(tgt)] for tgt in target_string_list]
    preds_word_tokens_no_tf = [tokenize_for_bleu(pred) for pred in pred_string_list]
    preds_word_tokens_tf = [tokenize_for_bleu(pred) for pred in pred_string_list_previous]

    mean_sent_bleu_scores = []
    mean_sent_bleu_scores_tf = []

    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(
            references_word_tokens,
            preds_word_tokens_no_tf,
            weights=weight,
            smoothing_function=smoothing
        )
        corpus_bleu_score_previous = corpus_bleu(
            references_word_tokens,
            preds_word_tokens_tf,
            weights=weight,
            smoothing_function=smoothing
        )
        corpus_bleu_scores.append(corpus_bleu_score)
        corpus_bleu_scores_previous.append(corpus_bleu_score_previous)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print(f'corpus BLEU-{len(list(weight))} score with tf:', corpus_bleu_score_previous)
        
        # Calculate Mean Sentence BLEU
        no_tf_sent = []
        tf_sent = []
        for tgt_toks, pred_toks, pred_tf_toks in zip(references_word_tokens, preds_word_tokens_no_tf, preds_word_tokens_tf):
            if len(pred_toks) == 0:
                b_no_tf = 0.0
            else:
                b_no_tf = sentence_bleu(tgt_toks, pred_toks, weights=weight, smoothing_function=smoothing)
            
            if len(pred_tf_toks) == 0:
                b_tf = 0.0
            else:
                b_tf = sentence_bleu(tgt_toks, pred_tf_toks, weights=weight, smoothing_function=smoothing)
                
            no_tf_sent.append(b_no_tf)
            tf_sent.append(b_tf)
            
        mean_sent_bleu_scores.append(float(np.mean(no_tf_sent)))
        mean_sent_bleu_scores_tf.append(float(np.mean(tf_sent)))
        print(f'mean sentence BLEU-{len(list(weight))} score:', mean_sent_bleu_scores[-1])
        print(f'mean sentence BLEU-{len(list(weight))} score with tf:', mean_sent_bleu_scores_tf[-1])


    """ calculate sacre bleu score """
    
    reference_list = [[item] for item in target_string_list]

    #print(f'ref: {reference_list}')
    #print(f'pred: {prediction_list}')
    sacre_blue = metric.compute(predictions=pred_string_list, references=reference_list)
    sacre_blue_previous = metric.compute(predictions=pred_string_list_previous, references=reference_list)
    print("sacreblue score: ", sacre_blue, '\n')
    print("sacreblue score with tf: ", sacre_blue_previous)


    print()
    """ calculate rouge score """
    rouge = Rouge()
    
    # pred_string_list = [item for sublist in pred_string_list for item in sublist]
    # pred_string_list = [item for sublist in pred_string_list for item in sublist]
    # pred_string_list_previous = [item for sublist in pred_string_list_previous for item in sublist]
    # rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg = True, ignore_empty=True)
    # rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    # print('rouge_scores: ', rouge_scores)
    # print('rouge_scores with tf:', rouge_scores_previous)

    # rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    # print('rouge_scores', rouge_scores)
    # print('previous rouge_scores', rouge_scores_previous)

    try:
        rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores = 'Hypothesis is empty'

    try:
        rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores_previous = 'Hypothesis is empty'
    print()


    print()
    """ calculate WER score """
    #wer = WordErrorRate()
    wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    wer_scores_previous = wer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("WER score:", wer_scores)
    print("WER score with tf:", wer_scores_previous)
    

    """ calculate CER score """
    cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    cer_scores_previous = cer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("CER score:", cer_scores)
    print("CER score with tf:", cer_scores_previous)

    """ calculate BERTScore """
    try:
        bertscore_results = bertscore_metric.compute(predictions=pred_string_list, references=target_string_list, lang="en")
        bertscore_f1 = float(np.mean(bertscore_results['f1']))
        bertscore_results_previous = bertscore_metric.compute(predictions=pred_string_list_previous, references=target_string_list, lang="en")
        bertscore_f1_previous = float(np.mean(bertscore_results_previous['f1']))
        print(f"BERTScore F1: {bertscore_f1:.4f}")
        print(f"BERTScore F1 with tf: {bertscore_f1_previous:.4f}")
    except Exception as e:
        # BERTScore is optional; if the tokenizer/model interface breaks for your env,
        # keep the rest of evaluation metrics (BLEU/ROUGE/WER/CER) working.
        print(f"[WARN] BERTScore failed, skipping. Error: {type(e).__name__}: {e}")
        bertscore_f1 = 0.0
        bertscore_f1_previous = 0.0


    end_time = time.time()
    print(f"Evaluation took {(end_time-start_time)/60} minutes to execute.")

     # score_results (only fix teacher-forcing)
    file_content = [
    f'mean_sentence_bleu_score = {mean_sent_bleu_scores}',
    f'corpus_bleu_score = {corpus_bleu_scores}',
    f'sacre_blue_score = {sacre_blue}',
    f'rouge_scores = {rouge_scores}',
    f'wer_scores = {wer_scores}',
    f'cer_scores = {cer_scores}',
    f'bertscore_f1 = {bertscore_f1}',
    f'mean_sentence_bleu_score_with_tf = {mean_sent_bleu_scores_tf}',
    f'corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}',
    f'sacre_blue_score_with_tf = {sacre_blue_previous}',
    f'rouge_scores_with_tf = {rouge_scores_previous}',
    f'wer_scores_with_tf = {wer_scores_previous}',
    f'cer_scores_with_tf = {cer_scores_previous}',
    f'bertscore_f1_with_tf = {bertscore_f1_previous}',
    ]
    
    with open(score_results, "w", encoding='utf-8') as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")

    return {
        'corpus_bleu': corpus_bleu_scores,
        'corpus_bleu_tf': corpus_bleu_scores_previous,
        'sacrebleu': sacre_blue,
        'sacrebleu_tf': sacre_blue_previous,
        'rouge': rouge_scores,
        'rouge_tf': rouge_scores_previous,
        'wer': wer_scores,
        'wer_tf': wer_scores_previous,
        'cer': cer_scores,
        'cer_tf': cer_scores_previous,
        'bertscore_f1': bertscore_f1,
        'bertscore_f1_tf': bertscore_f1_previous,
        'no_tf_per_sample': no_tf_scores,
        'tf_per_sample': tf_scores,
    }



def generate_ieee_figures(results, model_name, task_name, save_dir):
    """Generate publication-quality figures for IEEE conference papers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # IEEE two-column format styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Color palette (colorblind-friendly)
    C_NO_TF = '#2166AC'   # blue
    C_TF = '#B2182B'      # red
    C_ACCENT = '#4DAF4A'  # green

    # ===== Figure 1: BLEU Scores Comparison =====
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bleu_labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    bleu_no_tf = [s * 100 for s in results['corpus_bleu']]
    bleu_tf = [s * 100 for s in results['corpus_bleu_tf']]
    x = np.arange(len(bleu_labels))
    w = 0.32
    bars1 = ax.bar(x - w/2, bleu_no_tf, w, label='Without TF', color=C_NO_TF, edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, bleu_tf, w, label='With TF', color=C_TF, edgecolor='white', linewidth=0.5, zorder=3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'Corpus BLEU Scores — {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(bleu_labels)
    ax.legend(framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0, max(bleu_tf) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'bleu_scores.png'))
    fig.savefig(os.path.join(save_dir, 'bleu_scores.pdf'))
    plt.close(fig)
    print(f'  Saved BLEU figure to {save_dir}')

    # ===== Figure 2: ROUGE Scores Comparison =====
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rouge_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    rouge_keys = ['rouge-1', 'rouge-2', 'rouge-l']
    rouge_no_tf_f = []
    rouge_tf_f = []
    for k in rouge_keys:
        if isinstance(results['rouge'], dict):
            rouge_no_tf_f.append(results['rouge'][k]['f'] * 100)
        else:
            rouge_no_tf_f.append(0)
        if isinstance(results['rouge_tf'], dict):
            rouge_tf_f.append(results['rouge_tf'][k]['f'] * 100)
        else:
            rouge_tf_f.append(0)
    x = np.arange(len(rouge_labels))
    bars1 = ax.bar(x - w/2, rouge_no_tf_f, w, label='Without TF', color=C_NO_TF, edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, rouge_tf_f, w, label='With TF', color=C_TF, edgecolor='white', linewidth=0.5, zorder=3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title(f'ROUGE F1-Scores — {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(rouge_labels)
    ax.legend(framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0, max(rouge_tf_f + rouge_no_tf_f) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'rouge_scores.png'))
    fig.savefig(os.path.join(save_dir, 'rouge_scores.pdf'))
    plt.close(fig)
    print(f'  Saved ROUGE figure to {save_dir}')

    # ===== Figure 3: WER & CER Comparison =====
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    err_labels = ['WER', 'CER']
    err_no_tf = [results['wer'] * 100, results['cer'] * 100]
    err_tf = [results['wer_tf'] * 100, results['cer_tf'] * 100]
    x = np.arange(len(err_labels))
    bars1 = ax.bar(x - w/2, err_no_tf, w, label='Without TF', color=C_NO_TF, edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, err_tf, w, label='With TF', color=C_TF, edgecolor='white', linewidth=0.5, zorder=3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Error Rates — {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(err_labels)
    ax.legend(framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0, max(err_no_tf) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'error_rates.png'))
    fig.savefig(os.path.join(save_dir, 'error_rates.pdf'))
    plt.close(fig)
    print(f'  Saved error rates figure to {save_dir}')

    # ===== Figure 4: Per-Sample BLEU-1 Distribution (Histogram) =====
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    no_tf_bleu1 = [s['bleu1_score'] for s in results['no_tf_per_sample']]
    tf_bleu1 = [s['bleu1_score'] for s in results['tf_per_sample']]
    bins = np.linspace(0, 1, 25)
    axes[0].hist(no_tf_bleu1, bins=bins, color=C_NO_TF, edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)
    axes[0].set_title('Without Teacher Forcing')
    axes[0].set_xlabel('Sentence-Level BLEU-1')
    axes[0].set_ylabel('Number of Samples')
    axes[0].axvline(np.mean(no_tf_bleu1), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(no_tf_bleu1):.3f}')
    axes[0].legend(framealpha=0.9, edgecolor='gray')
    axes[1].hist(tf_bleu1, bins=bins, color=C_TF, edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)
    axes[1].set_title('With Teacher Forcing')
    axes[1].set_xlabel('Sentence-Level BLEU-1')
    axes[1].axvline(np.mean(tf_bleu1), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(tf_bleu1):.3f}')
    axes[1].legend(framealpha=0.9, edgecolor='gray')
    fig.suptitle(f'Per-Sample BLEU-1 Distribution — {model_name}', fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'bleu1_distribution.png'))
    fig.savefig(os.path.join(save_dir, 'bleu1_distribution.pdf'))
    plt.close(fig)
    print(f'  Saved BLEU-1 distribution to {save_dir}')

    # ===== Figure 5: Comprehensive Radar / Summary Bar Chart =====
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    summary_labels = ['BLEU-1', 'BLEU-4', 'ROUGE-1\nF1', 'ROUGE-L\nF1', 'SacreBLEU', '1-WER', '1-CER']
    sacre_no_tf = results['sacrebleu']['score'] if isinstance(results['sacrebleu'], dict) else 0
    sacre_tf = results['sacrebleu_tf']['score'] if isinstance(results['sacrebleu_tf'], dict) else 0
    r1_no_tf = results['rouge']['rouge-1']['f'] * 100 if isinstance(results['rouge'], dict) else 0
    rl_no_tf = results['rouge']['rouge-l']['f'] * 100 if isinstance(results['rouge'], dict) else 0
    r1_tf = results['rouge_tf']['rouge-1']['f'] * 100 if isinstance(results['rouge_tf'], dict) else 0
    rl_tf = results['rouge_tf']['rouge-l']['f'] * 100 if isinstance(results['rouge_tf'], dict) else 0
    vals_no_tf = [
        results['corpus_bleu'][0] * 100,
        results['corpus_bleu'][3] * 100,
        r1_no_tf, rl_no_tf,
        sacre_no_tf,
        (1 - results['wer']) * 100,
        (1 - results['cer']) * 100,
    ]
    vals_tf = [
        results['corpus_bleu_tf'][0] * 100,
        results['corpus_bleu_tf'][3] * 100,
        r1_tf, rl_tf,
        sacre_tf,
        (1 - results['wer_tf']) * 100,
        (1 - results['cer_tf']) * 100,
    ]
    x = np.arange(len(summary_labels))
    bars1 = ax.bar(x - w/2, vals_no_tf, w, label='Without TF', color=C_NO_TF, edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, vals_tf, w, label='With TF', color=C_TF, edgecolor='white', linewidth=0.5, zorder=3)
    ax.set_ylabel('Score (%)')
    ax.set_title(f'Overall Performance Summary — {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_labels, fontsize=7)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0, max(max(vals_no_tf), max(vals_tf)) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'overall_summary.png'))
    fig.savefig(os.path.join(save_dir, 'overall_summary.pdf'))
    plt.close(fig)
    print(f'  Saved overall summary figure to {save_dir}')

    print(f'\n[INFO] All figures saved to: {save_dir}')


if __name__ == '__main__': 
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name', type=str, default=None, help='Base name for result files (for ablation runs)')
    args, _ = parser.parse_known_args()

    config = get_config('eval_decoding')
    test_input = config['test_input']
    print("test_input is:", test_input)
    train_input = config['train_input']
    print("train_input is:", train_input)
    training_config = json.load(open(config['config_path']))

    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']
    model_name = training_config['model_name']

    RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Result')

    # Use ablation-specific result name if provided
    if args.result_name is not None:
        output_all_results_path = os.path.join(RESULT_DIR, f'{args.result_name}_all_results.txt')
        score_results = os.path.join(RESULT_DIR, f'{args.result_name}_score.txt')
    else:
        if test_input == 'EEG' and train_input=='EEG':
            print("EEG and EEG")
            output_all_results_path = os.path.join(RESULT_DIR, 'ZuCo_dataset_v1_v2', 'Translated_Text', f'{task_name}-{model_name}.txt')
            score_results = os.path.join(RESULT_DIR, 'ZuCo_dataset_v1_v2', 'Score', f'{task_name}-{model_name}.txt')
        else:
            output_all_results_path = os.path.join(RESULT_DIR, 'different_all_result', f'{model_name}-Random_test({test_input})-all_decoding_results.txt')
            score_results = os.path.join(RESULT_DIR, 'different_score_result', f'{model_name}-Random_test({test_input}).txt')

    # Ensure result directories exist
    os.makedirs(os.path.dirname(output_all_results_path), exist_ok=True)
    os.makedirs(os.path.dirname(score_results), exist_ok=True)


    ''' set random seeds '''
    seed_val = 20 #500
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = 0
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    # task_name = 'task1_task2_task3'

    ''' set up dataloader '''
    PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'pickle_file')
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = os.path.join(PICKLE_DIR, 'task1-SR-dataset.pickle')
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = os.path.join(PICKLE_DIR, 'task2-NR-dataset.pickle')
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = os.path.join(PICKLE_DIR, 'task2-TSR-2.0-dataset.pickle')
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = os.path.join(PICKLE_DIR, 'task2-TSR-2.0-dataset.pickle')
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()
    
    if model_name in ['BrainTranslator','BrainTranslatorNaive','R1Translator','EEGConformer']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    
    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        # tokenizer.set_prefix_tokens(language='english')

    # test dataset
    feature_level = training_config.get('feature_level', 'word')
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=test_input, feature_level=feature_level)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=0)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = config['checkpoint_path']
    
    if model_name == 'BrainTranslator':
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    elif model_name == 'BrainTranslatorNaive':
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslatorNaive(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'BertGeneration':
        pretrained = BertGenerationDecoder.from_pretrained('google-bert/bert-large-uncased', is_decoder = True)
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
        
    elif model_name == 'PegasusTranslator':
        pretrained = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    elif model_name == 'T5Translator':
        pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    elif model_name == 'R1Translator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large', torch_dtype=torch.float32)
        pretrained = pretrained.float()  # Ensure all parameters are float32
        model = R1Translator(
                pretrained,
                in_feature = 105*len(bands_choice),
                decoder_embedding_size = pretrained.config.d_model,
                rnn_hidden_size = 256,
                num_rnn_layers =2
            )
    elif model_name == 'EEGConformer':
        ablate_sba = training_config.get('ablate_sba', False)
        ablate_multiscale = training_config.get('ablate_multiscale', False)
        ablate_cab = training_config.get('ablate_cab', False)
        ablate_label_smoothing = training_config.get('ablate_label_smoothing', False)
        
        conv_kernels_str = training_config.get('conv_kernels', '1,3,5')
        if isinstance(conv_kernels_str, str):
            conv_kernels = tuple(map(int, conv_kernels_str.split(',')))
        else:
            conv_kernels = (1, 3, 5)
        
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large', torch_dtype=torch.float32)
        pretrained = pretrained.float()
        pretrained.config.label_smoothing = 0.0 if ablate_label_smoothing else 0.1
        model = EEGConformer(
                pretrained,
                in_feature=105*len(bands_choice),
                decoder_embedding_size=pretrained.config.d_model,
                n_bands=len(bands_choice),
                n_electrodes=105,
                conv_channels=105*len(bands_choice),
                rnn_hidden_size=512,
                num_rnn_layers=3,
                n_bridge_heads=8,
                dropout=0.3,
                ablate_sba=ablate_sba,
                ablate_multiscale=ablate_multiscale,
                ablate_cab=ablate_cab,
                conv_kernels=conv_kernels
            )

    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if "multi_scale_conv.conv1." in k: k = k.replace("conv1.", "convs.0.")
        elif "multi_scale_conv.conv3." in k: k = k.replace("conv3.", "convs.1.")
        elif "multi_scale_conv.conv5." in k: k = k.replace("conv5.", "convs.2.")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    '''
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path))
    '''

    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    results = eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path, score_results=score_results)

    ''' generate IEEE-quality figures '''
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', f'{task_name}_{model_name}')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generate_ieee_figures(results, model_name, task_name, FIGURES_DIR)
