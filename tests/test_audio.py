from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from secure_audio.audio import AudioError, augment, fix_length, load_audio, mel_spectrogram, mfcc_statistics


def test_resample_downmix(tmp_path: Path, cfg: dict) -> None:
    path = tmp_path / "stereo.wav"
    wave = .2 * np.sin(2 * np.pi * 300 * np.arange(8000) / 8000)
    sf.write(path, np.stack([wave, wave], axis=1), 8000)
    loaded = load_audio(path, cfg["audio"])
    assert loaded.shape == (16000,)
    assert loaded.dtype == np.float32
    assert np.isfinite(loaded).all()


@pytest.mark.parametrize("case", ["silence", "short", "corrupt", "nonfinite"])
def test_reject_invalid(tmp_path: Path, cfg: dict, case: str) -> None:
    path = tmp_path / "input.wav"
    if case == "corrupt":
        path.write_bytes(b"not audio")
    else:
        samples = {"silence": np.zeros(16000), "short": np.ones(10),
                   "nonfinite": np.full(16000, np.nan)}[case]
        sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(AudioError):
        load_audio(path, cfg["audio"])


def test_crop_padding_and_seed(cfg: dict) -> None:
    wave = np.arange(12000, dtype=np.float32)
    assert np.array_equal(fix_length(wave, cfg["audio"]), wave[2000:10000])
    a = fix_length(wave, cfg["audio"], np.random.default_rng(5))
    b = fix_length(wave, cfg["audio"], np.random.default_rng(5))
    assert np.array_equal(a, b)
    padded = fix_length(wave[:200], cfg["audio"])
    assert len(padded) == 8000 and np.all(padded[200:] == 0)


@pytest.mark.parametrize("kind", ["mel", "logmel"])
def test_feature_shapes(cfg: dict, kind: str) -> None:
    cfg["features"]["spectrogram"] = kind
    wave = np.random.default_rng(1).normal(0, .1, 8000).astype(np.float32)
    spec = mel_spectrogram(wave, cfg)
    assert spec.shape == (1, 64, 51)
    assert spec.dtype == np.float32 and np.isfinite(spec).all()
    features = mfcc_statistics(wave, cfg)
    assert features.shape == (120,)
    assert np.isfinite(features).all()


def test_mel_logmel_are_distinct(cfg: dict) -> None:
    wave = np.random.default_rng(1).normal(0, .1, 8000).astype(np.float32)
    cfg["features"]["spectrogram"] = "mel"
    mel = mel_spectrogram(wave, cfg)
    cfg["features"]["spectrogram"] = "logmel"
    logmel = mel_spectrogram(wave, cfg)
    np.testing.assert_allclose(logmel, 10 * np.log10(np.maximum(mel, 1e-10)), atol=1e-5)


def test_augmentation_reproducible(cfg: dict) -> None:
    cfg["augmentation"]["enabled"] = True
    wave = np.ones(8000, dtype=np.float32) * .1
    a = augment(wave, 16000, cfg["augmentation"], np.random.default_rng(5))
    b = augment(wave, 16000, cfg["augmentation"], np.random.default_rng(5))
    assert np.array_equal(a, b) and not np.array_equal(a, wave)
