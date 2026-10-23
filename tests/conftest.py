"""Small generated audio fixtures; never stand in for real benchmark results."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from secure_audio.common import read_config, save_json
from secure_audio.data import audit, protocol_path


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    config = copy.deepcopy(read_config("configs/default.yaml"))
    config["audio"]["seconds"] = .5
    config["output_root"] = str(tmp_path / "artifacts")
    config["cache_root"] = str(tmp_path / "cache")
    config["data_root"] = str(tmp_path / "LA")
    config["baseline"].update(svm_c=[1.0], boosting_leaves=[7], boosting_iterations=5)
    config["deep"].update(epochs=1, batch_size=4, threads=2, device="cpu")
    return config


@pytest.fixture
def tiny_dataset(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Protocol-shaped fixtures with disjoint speakers, IDs and decoded content."""
    import secure_audio.data as data
    root = Path(cfg["data_root"])
    for split_index, split in enumerate(data.SPLITS):
        monkeypatch.setitem(data.COUNTS, split, (4, 4))
        protocol = protocol_path(root, split)
        protocol.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        prefix = {"train": "T", "dev": "D", "eval": "E"}[split]
        for index in range(8):
            label = index % 2
            uid = f"LA_{prefix}_{index:07d}"
            attack = "-" if label else ("A07" if split == "eval" else "A01")
            lines.append(f"speaker_{split}_{index} {uid} - {attack} {'bonafide' if label else 'spoof'}")
            path = root / f"ASVspoof2019_LA_{split}" / "flac" / f"{uid}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            t = np.arange(8000) / 16000
            wave = .2 * np.sin(2 * np.pi * (300 + 1100 * (1-label) + index + split_index*.25) * t)
            sf.write(path, wave, 16000)
        protocol.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(root / "audit.json", audit(root, content_hash=True))
    return cfg
