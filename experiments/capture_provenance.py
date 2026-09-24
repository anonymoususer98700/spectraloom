#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PICKLES = REPO / "EEG-To-text" / "Data" / "pickle_file"
FILES = (
    "task1-SR-dataset.pickle",
    "task2-NR-dataset.pickle",
    "task3-TSR-dataset.pickle",
    "task2-TSR-2.0-dataset.pickle",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    import numpy
    import torch
    import transformers

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "provenance.json"
    datasets = {}
    for name in FILES:
        path = PICKLES / name
        if not path.is_file():
            raise FileNotFoundError(path)
        datasets[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    payload = {
        "captured_unix_time": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "gpu_driver": driver,
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
