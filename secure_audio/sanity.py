"""Diagnostic checks with explicit pass criteria and no benchmark claims."""
from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch import nn

from .audio import AudioError, fix_length, load_audio, mel_spectrogram
from .common import seed_all
from .data import read_protocol
from .models import build_mobile, optimization_step, resolve_device


def overfit(cfg: dict, *, real_data: bool, steps: int = 200) -> dict:
    """Memorize 8 fixed inputs; require >=95% train accuracy and BCE below 0.15.

    The synthetic mode verifies mechanics only. The real mode uses four trials
    per class from the official training protocol, with no dev/eval involvement.
    """
    seed_all(cfg["seeds"][0], cfg["deep"]["threads"])
    cfg = copy.deepcopy(cfg)
    cfg["audio"]["seconds"] = min(cfg["audio"]["seconds"], 1.0)
    cfg["augmentation"]["enabled"] = False
    inputs, targets = [], []
    if real_data:
        records = read_protocol(Path(cfg["data_root"]), "train")
        chosen = [r for label in (0, 1) for r in [r for r in records if r.label == label][:4]]
        if len(chosen) != 8:
            raise ValueError("Need four real training trials per class")
        for r in chosen:
            inputs.append(mel_spectrogram(fix_length(load_audio(r.path, cfg["audio"]), cfg["audio"]), cfg))
            targets.append(r.label)
    else:
        rng = np.random.default_rng(42)
        t = np.arange(round(cfg["audio"]["sample_rate"] * cfg["audio"]["seconds"])) / cfg["audio"]["sample_rate"]
        for label in (0, 1):
            for _ in range(4):
                wave = (0.2 * np.sin(2 * np.pi * (300 if label else 2300) * t) + rng.normal(0, .005, len(t))).astype(np.float32)
                inputs.append(mel_spectrogram(wave, cfg))
                targets.append(label)
    device = resolve_device(cfg)
    model = build_mobile().to(device)
    x = torch.from_numpy(np.stack(inputs)).to(device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    print(f"Tiny-set overfit: device={device}, clips={len(targets)}, steps={steps}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()
    initial = None
    for step in range(steps):
        model.train()
        loss = optimization_step(model, optimizer, criterion, x, y)
        if initial is None:
            initial = loss
        if (step + 1) % 25 == 0:
            print(f"Overfit step {step+1}/{steps}: training BCE={loss:.5f}", flush=True)
    # Calibrate running BN on the tiny diagnostic set, then test in inference mode.
    # No such recalibration is done on held-out data in the real experiment.
    model.train()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        model.eval()
        logits = model(x).flatten()
        final_loss = float(criterion(logits, y))
        accuracy = float(((logits >= 0) == y.bool()).float().mean())
    return {"mode": "real_train_subset" if real_data else "synthetic_mechanics_only", "steps": steps,
            "device": str(device),
            "n": len(y), "initial_train_bce": initial, "final_inference_bce": final_loss,
            "train_accuracy": accuracy, "passed": accuracy >= .95 and final_loss < .15,
            "uses_tiny_set_bn_calibration": True}


def shuffled_labels(cfg: dict, *, real_data: bool) -> dict:
    """Shuffle training labels only; evaluate balanced accuracy on untouched dev.

    Synthetic mode is a negative-control mechanics test, not evidence about LA.
    Real mode extracts all train/dev MFCCs and repeats the control for all seeds.
    No threshold or model tuning is allowed for this diagnostic.
    """
    from .cache import cache_features
    from .train import validated_splits
    seed_all(cfg["seeds"][0], cfg["deep"]["threads"])
    if real_data:
        splits = validated_splits(cfg)
        x_train, x_dev = [cache_features(splits[s], cfg, "mfcc") for s in ("train", "dev")]
        y_train, y_dev = [np.array([r.label for r in splits[s]]) for s in ("train", "dev")]
    else:
        rng = np.random.default_rng(99)
        x_train, x_dev = rng.normal(size=(1000, 20)), rng.normal(size=(1000, 20))
        y_train, y_dev = (x_train[:, 0] > 0).astype(int), (x_dev[:, 0] > 0).astype(int)
    results = {"svm": [], "boosting": []}
    with threadpool_limits(limits=cfg["deep"]["threads"]):
        for seed in cfg["seeds"]:
            shuffled = np.random.default_rng(seed).permutation(y_train)
            models = {"boosting": HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=7,
                      class_weight="balanced", early_stopping=False, random_state=seed),
                      "svm": make_pipeline(StandardScaler(), SVC(C=1.0, class_weight="balanced", random_state=seed))}
            for name, model in models.items():
                started = time.perf_counter()
                print(f"Shuffled {name} seed={seed}: fitting {len(y_train)} MFCC trials on CPU", flush=True)
                model.fit(x_train, shuffled)
                value = float(balanced_accuracy_score(y_dev, model.predict(x_dev)))
                results[name].append(value)
                print(f"Shuffled {name} seed={seed}: balanced accuracy={value:.4f}; {time.perf_counter()-started:.1f}s", flush=True)
    summaries = {name: {"balanced_accuracy": values, "mean": float(np.mean(values)),
                       "sample_std": float(np.std(values, ddof=1)),
                       "passed": .45 <= float(np.mean(values)) <= .55}
                 for name, values in results.items()}
    return {"mode": "real_train_dev_mfcc" if real_data else "synthetic_mechanics_only",
            "seeds": cfg["seeds"], "models": summaries, "expected_band": [.45, .55],
            "passed": all(value["passed"] for value in summaries.values())}


def shuffled_deep(cfg: dict, *, real_data: bool) -> dict:
    """Fixed-budget MobileNet negative control; no dev model or threshold tuning.

    Permute train labels only. Train for the predeclared epoch budget and score
    untouched dev exactly once using logit >= 0. Synthetic mode is deliberately
    small and tests mechanics; real mode uses the complete official train/dev.
    """
    from dataclasses import replace
    from torch.utils.data import TensorDataset
    from .cache import cache_features
    from .models import SpectrogramDataset, loader, predict_deep
    from .train import validated_splits
    cfg = copy.deepcopy(cfg)
    cfg["augmentation"]["enabled"] = False
    if real_data:
        splits = validated_splits(cfg)
        train, dev = splits["train"], splits["dev"]
        train_x = cache_features(train, cfg, "spectrogram")
        dev_x = cache_features(dev, cfg, "spectrogram")
        y_train, y_dev = [np.array([r.label for r in records]) for records in (train, dev)]
        epochs = cfg["deep"]["epochs"]
    else:
        # Shared learnable pattern, independent samples and balanced classes.
        rng = np.random.default_rng(82)
        y_train, y_dev = np.tile([0, 1], 64), np.tile([0, 1], 128)
        train_x = rng.normal(0, .2, (128, 1, 64, 51)).astype(np.float32)
        dev_x = rng.normal(0, .2, (256, 1, 64, 51)).astype(np.float32)
        train_x[:, :, :32] += y_train[:, None, None, None]
        dev_x[:, :, :32] += y_dev[:, None, None, None]
        epochs = 3
    values = []
    for seed in cfg["seeds"]:
        seed_all(seed, cfg["deep"]["threads"])
        permuted = np.random.default_rng(seed).permutation(y_train)
        if real_data:
            shuffled_records = [replace(r, label=int(y)) for r, y in zip(train, permuted, strict=True)]
            training = SpectrogramDataset(shuffled_records, cfg, cached=train_x)
            validation = SpectrogramDataset(dev, cfg, cached=dev_x)
        else:
            training = TensorDataset(torch.from_numpy(train_x), torch.tensor(permuted, dtype=torch.float32))
            validation = TensorDataset(torch.from_numpy(dev_x), torch.tensor(y_dev, dtype=torch.float32))
        batches = loader(training, cfg, seed, shuffle=True)
        device = resolve_device(cfg)
        model = build_mobile().to(device)
        print(f"Shuffled MobileNet seed={seed}: device={device}; batches per epoch={len(batches)}", flush=True)
        positive_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight), device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["deep"]["learning_rate"],
                                      weight_decay=cfg["deep"]["weight_decay"])
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            model.train()
            for batch, (x, y) in enumerate(batches):
                optimization_step(model, optimizer, criterion, x.to(device), y.to(device))
                if batch % 100 == 0:
                    print(f"Shuffled MobileNet seed={seed} epoch={epoch+1}/{epochs} batch={batch}/{len(batches)} device={device}", flush=True)
            print(f"Shuffled MobileNet seed={seed} epoch={epoch+1}/{epochs}: {time.perf_counter()-epoch_started:.1f}s", flush=True)
        scores = predict_deep(model, loader(validation, cfg, seed), device)
        values.append(float(balanced_accuracy_score(y_dev, scores >= 0)))
    mean = float(np.mean(values))
    return {"mode": "real_train_dev_mobile" if real_data else "synthetic_mechanics_only",
            "epochs": epochs, "fixed_threshold": 0.0, "seeds": cfg["seeds"],
            "balanced_accuracy": values, "mean": mean, "sample_std": float(np.std(values, ddof=1)),
            "expected_band": [.45, .55], "passed": .45 <= mean <= .55}


def edge_cases(cfg: dict) -> dict:
    """Verify silence/very-short rejection and finite white-noise spectrograms."""
    result = {}
    with tempfile.TemporaryDirectory() as folder:
        sr = cfg["audio"]["sample_rate"]
        cases = {"silence": np.zeros(sr), "very_short": np.ones(8) * .2,
                 "noise": np.random.default_rng(12).normal(0, .1, sr)}
        for name, wave in cases.items():
            path = Path(folder) / f"{name}.wav"
            sf.write(path, wave, sr, subtype="FLOAT")
            try:
                loaded = fix_length(load_audio(path, cfg["audio"]), cfg["audio"])
                finite = bool(np.isfinite(mel_spectrogram(loaded, cfg)).all())
                result[name] = {"rejected": False, "finite_features": finite,
                                "passed": name == "noise" and finite}
            except AudioError as exc:
                result[name] = {"rejected": True, "reason": str(exc), "passed": name != "noise"}
    result["passed"] = all(value["passed"] for value in result.values())
    return result
