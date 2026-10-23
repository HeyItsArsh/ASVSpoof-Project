import json
from pathlib import Path

import numpy as np
import pytest
import torch

from secure_audio.cache import cache_features
from secure_audio.common import save_json, seed_all
from secure_audio.data import read_protocol
from secure_audio.experiments import aggregate, freeze_experiments, verify_frozen
from secure_audio.inference import Detector
from secure_audio.models import SpectrogramDataset, build_mobile, loader, predict_deep
from secure_audio.train import evaluate_run, fit_baseline, fit_deep


def test_forward_and_seed(cfg: dict) -> None:
    seed_all(13, 2)
    a = build_mobile().eval()
    seed_all(13, 2)
    b = build_mobile().eval()
    wave = torch.randn(2, 1, 64, 51)
    with torch.no_grad():
        assert a(wave).shape == (2, 1)
        torch.testing.assert_close(a(wave), b(wave), rtol=0, atol=0)


def test_cache_dataset_and_invalidation(tiny_dataset: dict) -> None:
    cfg = tiny_dataset
    records = read_protocol(Path(cfg["data_root"]), "train")
    cache = cache_features(records, cfg, "spectrogram")
    dataset = SpectrogramDataset(records, cfg, cached=cache)
    torch.testing.assert_close(dataset[0][0], torch.from_numpy(np.array(cache[0])))
    assert dataset[0][1].item() == records[0].label
    cache_features(records, cfg, "spectrogram")
    assert len(list(Path(cfg["cache_root"]).glob("spectrogram*.npy"))) == 1
    cfg["audio"]["seconds"] = .6
    cache_features(records, cfg, "spectrogram")
    assert len(list(Path(cfg["cache_root"]).glob("spectrogram*.npy"))) == 2


@pytest.mark.parametrize("name", ["svm", "boosting"])
def test_baseline_roundtrip_and_aggregation(tiny_dataset: dict, name: str) -> None:
    cfg = tiny_dataset
    directories = [fit_baseline(cfg, name, seed) for seed in cfg["seeds"]]
    root = Path(cfg["output_root"])
    freeze_experiments(root, cfg["seeds"])
    for directory in directories:
        verify_frozen(directory)
        metrics = evaluate_run(directory)
        assert 0 <= metrics["eer"] <= 1
        assert metrics["attacks"]["A07"]["available"]
        assert not metrics["attacks"]["A01"]["available"]
        assert sum(map(sum, metrics["confusion_matrix"])) == 8
    result = aggregate(root, cfg["seeds"])
    assert len(result[name]["eval"]["eer"]["values"]) == 3
    meta = json.loads((directories[0] / "run.json").read_text())
    meta["threshold"] += 1
    save_json(directories[0] / "run.json", meta)
    with pytest.raises(ValueError, match="modified"):
        verify_frozen(directories[0])


def test_deep_training_checkpoint_and_inference(tiny_dataset: dict, tmp_path: Path) -> None:
    cfg = tiny_dataset
    directory = fit_deep(cfg, 13)
    run = json.loads((directory / "run.json").read_text())
    assert run["best_epoch"] == 1
    assert run["parameters"] < 3_000_000
    detector = Detector(directory)
    record = read_protocol(Path(cfg["data_root"]), "dev")[0]
    result = detector.score(record.path)
    assert result["status"] == "scored" and np.isfinite(result["bonafide_score"])
    silence = tmp_path / "silence.wav"
    import soundfile as sf
    sf.write(silence, np.zeros(16000), 16000)
    assert detector.score(silence)["status"] == "rejected"
    noise = tmp_path / "noise.wav"
    sf.write(noise, np.random.default_rng(1).normal(0, .1, 16000), 16000)
    noise_result = detector.score(noise)
    assert noise_result["status"] == "scored" and noise_result["noise_warning"]


def test_eval_requires_freeze(tiny_dataset: dict) -> None:
    directory = fit_baseline(tiny_dataset, "svm", 13)
    with pytest.raises(ValueError, match="Freeze first"):
        evaluate_run(directory)


def test_cli_resume_preserves_completed_run(tiny_dataset: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import yaml
    from secure_audio.__main__ import main
    from secure_audio.common import digest
    cfg = tiny_dataset
    directory = fit_baseline(cfg, "svm", 13)
    before = digest(directory / "model.joblib")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["secure_audio", "--config", str(config_path), "train", "--model", "svm", "--seed", "13", "--resume"])
    main()
    assert digest(directory / "model.joblib") == before


def test_incomplete_checkpoint_is_preserved(tiny_dataset: dict) -> None:
    from secure_audio.train import run_directory
    directory = run_directory(tiny_dataset, "svm", 13)
    directory.mkdir(parents=True)
    (directory / "model.joblib").write_bytes(b"interrupted example")
    fit_baseline(tiny_dataset, "svm", 13)
    assert (directory / "model.interrupted.joblib").read_bytes() == b"interrupted example"
