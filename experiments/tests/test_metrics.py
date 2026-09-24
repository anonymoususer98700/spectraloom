from spectraloom_experiments.metrics import compute_metrics, edit_distance, rouge_l_f1


def test_identical_predictions_are_perfect_on_core_metrics():
    result = compute_metrics(
        ["this is a sufficiently long exact matching test sentence"],
        ["this is a sufficiently long exact matching test sentence"],
    )
    assert result.corpus_bleu > 99
    assert result.rouge_l_f1 == 100
    assert result.wer == 0
    assert result.cer == 0


def test_edit_distance_and_empty_prediction():
    assert edit_distance(["a", "b"], ["a", "c"]) == 1
    assert rouge_l_f1("hello", "") == 0
    result = compute_metrics(["hello world"], [""])
    assert result.wer == 100
    assert result.empty_prediction_rate == 100
