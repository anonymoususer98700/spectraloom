from spectraloom_experiments.protocols import build_manifest, validate_manifest


def sent(text):
    return {"content": text, "word": [{"word_level_EEG": {}}]}


def test_sentence_groups_never_cross_phases_or_tasks():
    dataset_a = {"S1": [sent("shared"), sent("a")], "S2": [sent("shared"), sent("a")]}
    dataset_b = {"S3": [sent("shared"), sent("b")], "S4": [sent("shared"), sent("b")]}
    rows = build_manifest({"zuco1_sr": dataset_a, "zuco1_nr": dataset_b}, seed=7)
    assert validate_manifest(rows)["sentence_overlap_count"] == 0
    phases = {row["phase"] for row in rows if row["reference"] == "shared"}
    assert len(phases) == 1


def test_loso_unseen_sentence_excludes_held_subject_from_training():
    dataset = {
        "S1": [sent(str(i)) for i in range(20)],
        "S2": [sent(str(i)) for i in range(20)],
    }
    rows = build_manifest(
        {"zuco1_sr": dataset}, protocol="loso_unseen_sentence", held_out_subject="S2", seed=3
    )
    assert all(row["subject"] != "S2" for row in rows if row["phase"] in {"train", "dev"})
    assert all(row["subject"] == "S2" for row in rows if row["phase"] == "test")

