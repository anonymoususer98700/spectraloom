from __future__ import annotations

import csv
import importlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import canonical_text, example_id, sentence_id
from .protocols import TASK_FILES, iter_records, load_task


def add_legacy_to_path(repo_root: str | Path) -> Path:
    legacy = Path(repo_root) / "EEG-To-text"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    return legacy


def tasks_from_legacy_name(name: str) -> list[str]:
    # Explicit precedence prevents task2 from silently meaning ZuCo 2.
    result = []
    lowered = name.lower()
    if "task1" in lowered:
        result.append("zuco1_sr")
    if "task2" in lowered and "tasknrv2" not in lowered:
        result.append("zuco1_nr")
    if "task3" in lowered:
        result.append("zuco1_tsr")
    if "tasknrv2" in lowered:
        result.append("zuco2_tsr")
    if not result:
        raise ValueError(f"Cannot resolve any dataset from legacy task name {name!r}")
    return result


def legacy_test_references(dataset: dict) -> set[str]:
    subjects = list(dataset.keys())
    first_subject = subjects[0]
    seen = set()
    ordered = []
    for sent_obj in dataset[first_subject]:
        if sent_obj is None:
            continue
        text = canonical_text(str(sent_obj.get("content", "")))
        if text and text not in seen:
            ordered.append(text)
            seen.add(text)
    dev_end = int(0.8 * len(ordered)) + int(0.1 * len(ordered))
    return set(ordered[dev_end:])


class LegacyMetaDataset:
    """Checkpoint-compatible legacy preprocessing with stable sample metadata."""

    def __init__(
        self,
        repo_root: str | Path,
        tasks: list[str],
        tokenizer,
        eeg_type: str = "GD",
        bands: list[str] | None = None,
        feature_level: str = "word",
        pickle_dir: str | Path | None = None,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        legacy = add_legacy_to_path(repo_root)
        data_module = importlib.import_module("Data")
        get_input_sample = data_module.get_input_sample
        self.inputs: list[dict] = []
        self.tasks = tasks
        bands = bands or ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
        pickle_dir = Path(pickle_dir or legacy / "Data" / "pickle_file")

        for task in tasks:
            dataset = load_task(pickle_dir, task)
            test_references = legacy_test_references(dataset)
            for row in iter_records(dataset, task):
                if row["reference"] not in test_references:
                    continue
                sample = get_input_sample(
                    row["sent_obj"],
                    tokenizer,
                    eeg_type=eeg_type,
                    bands=bands,
                    max_len=56,
                    test_input="EEG",
                    feature_level=feature_level,
                )
                if sample is None:
                    continue
                sample["meta"] = {key: value for key, value in row.items() if key != "sent_obj"}
                self.inputs.append(sample)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        return self.inputs[index]


def instantiate_legacy_model(repo_root: str | Path, config: dict, device):
    import torch
    from transformers import (
        BartForConditionalGeneration,
        PegasusForConditionalGeneration,
        T5ForConditionalGeneration,
    )

    add_legacy_to_path(repo_root)
    models = importlib.import_module("model_decoding")
    model_name = config["model_name"]
    bands = config.get("eeg_bands") or ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
    if model_name in {"EEGConformer", "EEGConformerStaticSBA", "EEG2TextFeatureAdapter", "BrainTranslator", "BrainTranslatorNaive", "R1Translator"}:
        pretrained = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large", torch_dtype=torch.float32
        ).float()
    elif model_name == "T5Translator":
        pretrained = T5ForConditionalGeneration.from_pretrained(
            "t5-large", torch_dtype=torch.float32
        ).float()
    elif model_name == "PegasusTranslator":
        pretrained = PegasusForConditionalGeneration.from_pretrained(
            "google/pegasus-xsum", torch_dtype=torch.float32
        ).float()
    else:
        raise ValueError(f"Unsupported checkpoint model {model_name!r}")

    if model_name in {"EEGConformer", "EEGConformerStaticSBA"}:
        kernels = config.get("conv_kernels", "3,5,7")
        if isinstance(kernels, str):
            kernels = tuple(int(v) for v in kernels.split(","))
        model_class = (
            importlib.import_module("model_static_sba").EEGConformerStaticSBA
            if model_name == "EEGConformerStaticSBA" else models.EEGConformer
        )
        model = model_class(
            pretrained,
            in_feature=105 * len(bands),
            decoder_embedding_size=pretrained.config.d_model,
            n_bands=len(bands),
            n_electrodes=105,
            conv_channels=105 * len(bands),
            rnn_hidden_size=512,
            num_rnn_layers=3,
            n_bridge_heads=8,
            dropout=float(config.get("dropout", 0.3)),
            ablate_sba=config.get("ablate_sba", False),
            ablate_multiscale=config.get("ablate_multiscale", False),
            ablate_cab=config.get("ablate_cab", False),
            conv_kernels=kernels,
        )
    elif model_name == "EEG2TextFeatureAdapter":
        model = importlib.import_module("model_feature_eeg2text").EEG2TextFeatureAdapter(
            pretrained, n_bands=len(bands)
        )
    elif model_name == "BrainTranslator":
        model = models.BrainTranslator(pretrained, 105 * len(bands), pretrained.config.d_model)
    elif model_name == "BrainTranslatorNaive":
        model = models.BrainTranslatorNaive(pretrained, 105 * len(bands), pretrained.config.d_model)
    elif model_name == "R1Translator":
        model = models.R1Translator(
            pretrained,
            in_feature=105 * len(bands),
            decoder_embedding_size=pretrained.config.d_model,
            rnn_hidden_size=256,
            num_rnn_layers=2,
        )
    elif model_name == "T5Translator":
        model = models.T5Translator(
            pretrained, 105 * len(bands), pretrained.config.d_model
        )
    elif model_name == "PegasusTranslator":
        model = models.BrainTranslator(
            pretrained, 105 * len(bands), pretrained.config.d_model
        )
    return model.to(device)


def tokenizer_name(model_name: str) -> str:
    return {
        "T5Translator": "t5-large",
        "PegasusTranslator": "google/pegasus-xsum",
    }.get(model_name, "facebook/bart-large")


def load_checkpoint(model, checkpoint: str | Path) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        key = key.replace("multi_scale_conv.conv1.", "multi_scale_conv.convs.0.")
        key = key.replace("multi_scale_conv.conv3.", "multi_scale_conv.convs.1.")
        key = key.replace("multi_scale_conv.conv5.", "multi_scale_conv.convs.2.")
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)


def seed_everything(seed: int) -> None:
    # PYTHONHASHSEED only takes effect at interpreter startup; the launcher sets
    # it as well. Keeping it here documents the seed in direct-script runs.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
