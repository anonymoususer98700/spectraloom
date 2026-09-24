"""Presentation-only helpers must preserve saved predictions verbatim."""
import importlib.util
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location(
    'paper_export', Path(__file__).resolve().parents[2] / 'scripts/update_paper_results.py')
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_highlight_preserves_raw_text_and_case():
    ref, pred = 'The film is good.', 'TheThe film was good.'
    a, b = paper.highlight_pair(ref, pred)
    unwrap = lambda s: re.sub(r'\\matchword\{([^}]*)\}', r'\1', s)
    assert unwrap(a) == ref
    assert unwrap(b) == pred
    assert r'\matchword{film}' in a and r'\matchword{film}' in b
    assert r'\matchword{TheThe}' not in b


def test_highlight_respects_repeated_word_counts():
    a, b = paper.highlight_pair('a a a', 'A')
    assert a.count(r'\matchword') == b.count(r'\matchword') == 1


def test_latex_escape_for_prediction_text():
    assert paper.latex_escape('50% & $x_1') == r'50\% \& \$x\_1'
