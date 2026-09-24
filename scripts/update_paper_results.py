"""Refresh paper tables from saved metrics only; never trains or evaluates models.

Run: python scripts/update_paper_results.py
Incomplete seed sets are displayed explicitly and excluded from ranking.
Checkpoint size counts include registered parameters, including unused template
layers, but exclude buffers and repeated tied embedding/output weights.
"""
import json
import statistics
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'experiments/outputs/training'
OUT = ROOT / 'paper_iclr/generated'
SEEDS = (42, 100, 312)
MODELS = {'braintranslator_bart': 'BrainTranslator',
          'eeg2text_feature': 'EEG2Text-FA',
          't5_large': 'T5-large', 'pegasus_xsum': 'Pegasus-XSum',
          'spectraloom': 'SpectraLoom'}
METRICS = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge_l_f1', 'wer', 'chrf']

def latex_escape(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                    '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in text)

def highlight_pair(reference, prediction):
    """Highlight case-insensitive, order-preserving word matches, without editing text."""
    pattern = re.compile(r"[\w]+(?:'[\w]+)?", re.UNICODE)
    a, b = list(pattern.finditer(reference)), list(pattern.finditer(prediction))
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)-1, -1, -1):
        for j in range(len(b)-1, -1, -1):
            dp[i][j] = (1+dp[i+1][j+1] if a[i].group().lower()==b[j].group().lower()
                        else max(dp[i+1][j], dp[i][j+1]))
    matched_a, matched_b = set(), set()
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i].group().lower() == b[j].group().lower():
            matched_a.add(i); matched_b.add(j); i+=1; j+=1
        elif dp[i+1][j] >= dp[i][j+1]:
            i+=1
        else:
            j+=1
    def render(text, matches, selected):
        chunks=[]; last=0
        for i, match in enumerate(matches):
            chunks.append(latex_escape(text[last:match.start()]))
            word=latex_escape(match.group())
            chunks.append(r'\matchword{'+word+'}' if i in selected else word)
            last=match.end()
        chunks.append(latex_escape(text[last:]))
        return ''.join(chunks)
    return render(reference,a,matched_a),render(prediction,b,matched_b)

def write_examples(snapshot):
    sys.path.insert(0, str(ROOT/'experiments/src'))
    from spectraloom_experiments.metrics import rouge_l_f1
    source=RUNS/'spectraloom_seed312/teacher_forced_test_predictions.csv'
    with source.open(encoding='utf-8', newline='') as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==1127 and all(r['seed']=='312' for r in rows)
    rows.sort(key=lambda r:(rouge_l_f1(r['reference'],r['prediction']),r['example_id']))
    chosen=[('Highest overlap',rows[-1]),('Median overlap',rows[len(rows)//2]),('Lowest overlap',rows[0])]
    lines=[r'\begin{tabular}{p{1.65cm}p{5.25cm}p{5.25cm}r}',r'\toprule',
           r'Case & Reference & Teacher-forced prediction & RL-F \\',r'\midrule']
    snapshot['teacher_forced_examples']=[]
    for label,row in chosen:
        ref,pred=highlight_pair(row['reference'],row['prediction'])
        score=100*rouge_l_f1(row['reference'],row['prediction'])
        lines.append(f'{label} & {ref} & {pred} & {score:.2f}'+r' \\[5pt]')
        snapshot['teacher_forced_examples'].append(dict(row,selection=label,rouge_l_f1=score))
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'teacher_forced_examples.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    import torch
    OUT.mkdir(exist_ok=True)
    snapshot = {'seeds': SEEDS, 'std': 'sample (ddof=1)', 'phases': {}, 'parameters': {}}
    for phase, filename in [('test', 'test_predictions'), ('zero_shot', 'zero_shot_predictions'),
                            ('teacher_forced', 'teacher_forced_test_predictions')]:
        data = {}
        for model in MODELS:
            rows = []
            for seed in SEEDS:
                path = RUNS / f'{model}_seed{seed}' / f'{filename}.metrics.json'
                if path.exists():
                    r = json.loads(path.read_text())
                    assert r['seed'] == seed
                    assert r['n'] == (6629 if phase == 'zero_shot' else 1127)
                    config = json.loads((path.parent / 'config.json').read_text())
                    complete = json.loads((path.parent / 'complete.json').read_text())
                    assert r['checkpoint_sha256'] == complete['best_checkpoint_sha256']
                    manifest = config['manifest_sha256']
                    if 'manifest_sha256' not in snapshot:
                        snapshot['manifest_sha256'] = manifest
                    assert manifest == snapshot['manifest_sha256']
                    if phase != 'teacher_forced':
                        assert r['decoding'] == dict(length_penalty=1.4, max_length=32,
                            min_new_tokens=0, no_repeat_ngram_size=2, num_beams=5,
                            repetition_penalty=1.5)
                    else:
                        assert r['evaluation_mode'] == 'teacher_forced_gold_prefix_argmax'
                    rows.append(r)
            data[model] = rows
        snapshot['phases'][phase] = data
        ranks = {}
        for metric in METRICS:
            values = {m: statistics.mean(r[metric] for r in rows)
                      for m, rows in data.items() if len(rows) == 3}
            ranks[metric] = sorted(set(values.values()), reverse=metric != 'wer')
        lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
                 r'Model & Seeds & B-1 & B-2 & B-3 & B-4 & RL-F & WER$\downarrow$ & chrF \\', r'\midrule']
        for model, rows in data.items():
            cells = [MODELS[model], str(len(rows))]
            for metric in METRICS:
                if not rows:
                    cells.append('XX'); continue
                vals = [r[metric] for r in rows]
                mean = statistics.mean(vals)
                cell = f'{mean:.3f}'
                if len(vals) > 1:
                    cell = (r'\shortstack{' + cell + r'\\{\scriptsize$\pm$'
                            + f'{statistics.stdev(vals):.3f}' + '}}')
                if len(rows) == 3:
                    rank = ranks[metric].index(mean)
                    if rank < 2:
                        cell = '\\' + ('best' if rank == 0 else 'second') + '{' + cell + '}'
                cells.append(cell)
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (OUT / f'{phase}_table.tex').write_text('\n'.join(lines)+'\n')
    for model in MODELS:
        checkpoint = RUNS / f'{model}_seed42' / 'best.pt'
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        total = custom = unused_template = 0
        excluded = []
        for name, tensor in state.items():
            if name.endswith(('final_logits_bias', '.pe', 'lm_head.weight',
                              'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight')):
                excluded.append(name)
                continue
            total += tensor.numel()
            if not name.startswith('pretrained.'):
                custom += tensor.numel()
            if name.startswith('additional_encoder_layer.'):
                unused_template += tensor.numel()
        snapshot['parameters'][model] = {'total': total, 'backbone': total-custom,
                                         'eeg_modules': custom, 'excluded_keys': excluded,
                                         'unused_template': unused_template,
                                         'active_total': total-unused_template}
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Local system & Backbone (M) & EEG modules (M) & Total (M)$\downarrow$ \\', r'\midrule']
    order = sorted(snapshot['parameters'], key=lambda m: snapshot['parameters'][m]['total'])
    for model in MODELS:
        p = snapshot['parameters'][model]
        total = f"{p['total']/1e6:.2f}"
        rank = order.index(model)
        if rank < 2:
            total = '\\' + ('best' if rank == 0 else 'second') + '{' + total + '}'
        lines.append(f"{MODELS[model]} & {p['backbone']/1e6:.2f} & {p['eeg_modules']/1e6:.2f} & {total}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'parameters_table.tex').write_text('\n'.join(lines)+'\n')
    write_examples(snapshot)
    (OUT / 'results_snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print('Updated paper tables and provenance snapshot:', OUT)

if __name__ == '__main__':
    main()
