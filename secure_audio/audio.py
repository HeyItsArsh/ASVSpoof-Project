"""One shared audio front end for all models; no silent discarding of trials."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


class AudioError(ValueError):
    """Audio cannot safely be classified; callers should request another recording."""


def load_audio(path: str | Path, cfg: dict[str, Any]) -> np.ndarray:
    """Read, downmix and resample; reject corrupt, nonfinite, short or silent input.

    Training and evaluation fail loudly on invalid inputs. Inference catches this
    exception and returns an explicit rejection, never an authentication success.
    """
    try:
        wave, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"Cannot read {path}: {exc}") from exc
    if wave.shape[0] / rate < cfg["min_seconds"]:
        raise AudioError("Clip is too short; record at least 0.1 s (or configured minimum)")
    if not np.isfinite(wave).all():
        raise AudioError("Non-finite audio")
    wave = wave.mean(axis=1)
    if rate != cfg["sample_rate"]:
        wave = librosa.resample(wave, orig_sr=rate, target_sr=cfg["sample_rate"])
    if float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) < cfg["silence_rms"]:
        raise AudioError("Silent or near-silent clip")
    return np.asarray(wave, dtype=np.float32)


def fix_length(wave: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero-pad short clips; center crop for evaluation, random crop for training."""
    size = round(cfg["sample_rate"] * cfg["seconds"])
    if len(wave) < size:
        return np.pad(wave, (0, size - len(wave))).astype(np.float32)
    start = int(rng.integers(len(wave) - size + 1)) if rng is not None else (len(wave) - size) // 2
    return wave[start:start + size].copy()


def augment(wave: np.ndarray, sample_rate: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Training-only gain and additive white noise with fixed seeded randomness."""
    del sample_rate  # retained in the interface for future time-domain transforms
    if not cfg["enabled"]:
        return wave
    gain = 10 ** (rng.uniform(*cfg["gain_db"]) / 20)
    snr = rng.uniform(*cfg["noise_snr_db"])
    noise = rng.normal(size=wave.shape)
    rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
    noise *= rms / (10 ** (snr / 20) * (np.sqrt(np.mean(noise ** 2)) + 1e-12))
    return np.clip(gain * (wave + noise), -1, 1).astype(np.float32)


def mel_spectrogram(wave: np.ndarray, cfg: dict) -> np.ndarray:
    """Return float32 [1, mel bins, frames]; log-Mel is fixed-reference power dB.

    No per-clip standardization: this preserves the Mel/log-Mel ablation.
    BatchNorm in MobileNet learns statistics from training batches only.
    """
    feature = cfg["features"]
    mel = librosa.feature.melspectrogram(
        y=wave, sr=cfg["audio"]["sample_rate"], n_fft=feature["n_fft"],
        hop_length=feature["hop_length"], n_mels=feature["n_mels"],
        power=2.0, center=True, pad_mode="constant")
    if feature["spectrogram"] == "logmel":
        mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    return mel[np.newaxis].astype(np.float32)


def mfcc_statistics(wave: np.ndarray, cfg: dict) -> np.ndarray:
    """Pool mean/std of MFCCs plus first/second deltas into 6*n_mfcc features."""
    f = cfg["features"]
    mfcc = librosa.feature.mfcc(y=wave, sr=cfg["audio"]["sample_rate"],
                              n_mfcc=f["n_mfcc"], n_fft=f["n_fft"],
                              hop_length=f["hop_length"], n_mels=f["n_mels"],
                              pad_mode="constant")
    stacked = np.concatenate([mfcc, librosa.feature.delta(mfcc, mode="nearest"),
                              librosa.feature.delta(mfcc, order=2, mode="nearest")])
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)]).astype(np.float32)
