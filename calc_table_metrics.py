import pandas as pd
from evaluate import load
from rouge import Rouge

pairs = [
    {
        "cat": "A",
        "t": "He served in the United States Senate as a Republican from Minnesota.",
        "p": "He was a member of the Republican Party, and served in the United States House of Representatives from 1917 to 1919."
    },
    {
        "cat": "B",
        "t": "The eldest son of William Parker, he was born in Norwich, in St. Saviour's parish.",
        "p": "He was born in New York City and raised in the Bronx."
    },
    {
        "cat": "C",
        "t": "``The Kid Stays in the Picture'' is a great story, terrifically told by the man who wrote it but this Cliff Notes edition is a cheat.",
        "p": "TheTheThe Great''ays in the Game'' is a film movie about andally funny by a talented who created it. also time Richard version is a little."
    },
    {
        "cat": "D",
        "t": "It's a head-turner -- thoughtfully written, beautifully read and, finally, deeply humanizing.",
        "p": "He's a greatyscing, a it crafted, well acted, well most, funny satisfying.."
    },
    {
        "cat": "E",
        "t": "The picture doesn't know it's a comedy.",
        "p": "He graduated from the University of Wisconsin-Madison with a degree in economics."
    }
]

rouge = Rouge()
bertscore = load("bertscore")

with open("table_metrics.txt", "w") as f:
    for pair in pairs:
        r_score = rouge.get_scores(pair['p'], pair['t'])[0]['rouge-l']['f']
        b_score = bertscore.compute(predictions=[pair['p']], references=[pair['t']], lang="en")['f1'][0]
        f.write(f"Cat {pair['cat']} | ROUGE-L: {r_score:.3f} | BERTScore: {b_score:.3f}\n")
print("Done writing to table_metrics.txt")
