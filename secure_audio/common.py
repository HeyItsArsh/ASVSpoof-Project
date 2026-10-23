"""Configuration, deterministic execution and artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_config(path: str | Path) -> dict[str, Any]:
    """Read YAML with paths interpreted relative to the working directory."""
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg["features"]["spectrogram"] not in {"mel", "logmel"}:
        raise ValueError("spectrogram must be mel or logmel")
    if cfg["audio"]["seconds"] <= 0 or cfg["audio"]["min_seconds"] <= 0:
        raise ValueError("Audio durations must be positive")
    return cfg


def seed_all(seed: int, threads: int = 4) -> None:
    """Seed Python, NumPy and Torch; reject nondeterministic Torch operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Avoid Windows physical-core discovery subprocesses and their locale output.
    os.environ["LOKY_MAX_CPU_COUNT"] = str(threads)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def save_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically; NaN metrics must be represented as null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False)
    # Unique names prevent concurrent writers from sharing an open .tmp file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as out:
        temporary = Path(out.name)
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    try:
        for attempt in range(12):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(.025 * 2**attempt, .5))
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
    """Return a streaming SHA-256 file digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
