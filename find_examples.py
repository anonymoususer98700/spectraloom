import pandas as pd
from evaluate import load
from rouge import Rouge

df = pd.read_csv('Result/ZuCo1_Legacy_BERTScore_all_results_predictions.csv')
targets = df['target_text'].tolist()
preds = df['predicted_text_no_tf'].tolist()

rouge = Rouge()
bertscore = load("bertscore")
bscores = bertscore.compute(predictions=preds, references=targets, lang="en")
bert_f1_scores = bscores['f1']

results = []
for i in range(len(targets)):
    try:
        r_score = rouge.get_scores(preds[i], targets[i])[0]['rouge-l']['f']
    except:
        r_score = 0.0
    results.append({
        'index': i,
        'target': targets[i],
        'pred': preds[i],
        'rouge_l': r_score,
        'bert_f1': bert_f1_scores[i],
        'target_len': len(targets[i].split())
    })

# Category D: Sentiment Reading (Movie reviews). Task 1 SR sentences are in the first 400. Let's filter for text containing "movie", "film", "story", "good", "bad", "funny".
cat_d = [r for r in results if r['bert_f1'] > 0.8 and r['rouge_l'] < 0.3 and any(w in r['target'].lower() for w in ['movie','film','story','good','bad','funny','character'])]
cat_d = sorted(cat_d, key=lambda x: x['bert_f1'], reverse=True)

# Category E: Failure cases. Length < 10, low BERT
cat_e = [r for r in results if r['target_len'] < 10 and r['bert_f1'] < 0.75]
cat_e = sorted(cat_e, key=lambda x: x['bert_f1'])

print("--- CAT D (Sentiment) ---")
for r in cat_d[:5]:
    print(f"R:{r['rouge_l']:.3f} | B:{r['bert_f1']:.3f}\nT: {r['target']}\nP: {r['pred']}\n")

print("--- CAT E (Failure) ---")
for r in cat_e[:5]:
    print(f"R:{r['rouge_l']:.3f} | B:{r['bert_f1']:.3f}\nT: {r['target']}\nP: {r['pred']}\n")
