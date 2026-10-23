"""Explicit invalid-input handling for individual clips and external probe sets."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch

from .audio import AudioError, fix_length, load_audio, mel_spectrogram, mfcc_statistics
from .common import digest, seed_all
from .models import build_mobile, resolve_device
from .train import baseline_scores


class Detector:
    """Load a trusted local artifact once, then score multiple clips.

    Joblib and Torch artifacts must come from this project, not untrusted users.
    This countermeasure never grants authentication or verifies speaker identity.
    """
    def __init__(self, directory: Path) -> None:
        """Restore weights, preprocessing configuration and the dev threshold."""
        self.meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.cfg = self.meta["config"]
        seed_all(self.meta["seed"], self.cfg["deep"]["threads"])
        self.device = resolve_device(self.cfg)
        if self.meta["model"] == "mobile":
            self.model = build_mobile().to(self.device)
            state = torch.load(directory / "model.pt", map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["state_dict"])
            self.model.eval()
        else:
            self.model = joblib.load(directory / "model.joblib")

    def score(self, path: Path) -> dict:
        """Return rejection or a research classification, with noise diagnostics.

        Spectral flatness is a heuristic warning, not a reliable speech detector.
        It is not used to remove trials or change the official benchmark metrics.
        """
        try:
            wave = load_audio(path, self.cfg["audio"])
        except AudioError as exc:
            return {"status": "rejected", "reason": str(exc), "file": str(path)}
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=wave, n_fft=512, hop_length=160)))
        wave = fix_length(wave, self.cfg["audio"])
        started = time.perf_counter()
        if self.meta["model"] == "mobile":
            batch = torch.from_numpy(mel_spectrogram(wave, self.cfg)[None]).to(self.device)
            with torch.inference_mode():
                score = float(self.model(batch).flatten().cpu()[0])
        else:
            score = float(baseline_scores(self.model, mfcc_statistics(wave, self.cfg)[None])[0])
        if not np.isfinite(score):
            return {"status": "rejected", "reason": "Non-finite model output", "file": str(path)}
        return {"status": "scored", "file": str(path), "bonafide_score": score,
                "prediction": "bonafide" if score >= self.meta["threshold"] else "spoof",
                "threshold_from_dev": self.meta["threshold"], "spectral_flatness": flatness,
                "noise_warning": flatness > .45,
                "authentication_decision": "not_provided_requires_speaker_and_speech_validation",
                "model_seconds": time.perf_counter() - started}


def external_probe(directory: Path, folder: Path) -> dict:
    """Score every generated WAV; report rejections separately from spoof detection.

    A spoof-only probe cannot yield EER, ROC or balanced accuracy. Never mix its
    clips into official splits or tune the model using the resulting scores.
    """
    detector = Detector(directory)
    files = sorted(folder.glob("*.wav"))
    if not files:
        raise ValueError(f"No WAV clips found in {folder}")
    scores = [detector.score(path) for path in files]
    valid = [r for r in scores if r["status"] == "scored"]
    return {"scope": "spoof_only_external_probe", "n_total": len(scores), "n_scored": len(valid),
            "manifest_sha256": digest(folder / "manifest.json") if (folder / "manifest.json").exists() else None,
            "n_rejected": len(scores) - len(valid), "eer": None,
            "spoof_detection_rate_scored": (sum(r["prediction"] == "spoof" for r in valid) / len(valid)) if valid else None,
            "clips": scores}
