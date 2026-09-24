import csv

from spectraloom_experiments.io import read_predictions
from spectraloom_experiments.stats import holm_adjust


def test_legacy_columns_are_canonicalized(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_text", "predicted_text_no_tf"])
        writer.writeheader()
        writer.writerow({"target_text": "  hello  world ", "predicted_text_no_tf": "hello"})
    rows = read_predictions(path)
    assert rows[0]["reference"] == "hello world"
    assert rows[0]["prediction"] == "hello"
    assert rows[0]["example_id"]
    assert rows[0]["sentence_id"]


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0 <= value <= 1 for value in adjusted)

