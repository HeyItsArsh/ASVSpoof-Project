"""Train-only fitting and dev-only selection; evaluation is a separate command."""
from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch import nn

from .cache import cache_features
from .common import digest, save_json, seed_all
from .data import COUNTS, SPLITS, assert_disjoint, protocol_path, read_protocol
from .metrics import dev_threshold, eer, write_evaluation
from .models import SpectrogramDataset, build_mobile, loader, optimization_step, predict_deep, resolve_device


def validated_splits(cfg: dict) -> dict:
    """Verify protocols and refuse fitting until the full data audit has passed."""
    import json
    root = Path(cfg["data_root"])
    splits = {s: read_protocol(root, s) for s in SPLITS}
    assert_disjoint(splits)
    receipt = root / "audit.json"
    if not receipt.exists():
        raise ValueError("Run check-data --content-hash before experiments")
    audit = json.loads(receipt.read_text(encoding="utf-8"))
    if not audit.get("decoded_content_checked"):
        raise ValueError("A decoded-content audit is required: check-data --content-hash")
    for split in SPLITS:
        counts = tuple(sum(r.label == label for r in splits[split]) for label in (1, 0))
        if counts != COUNTS[split] or digest(protocol_path(root, split)) != audit[split]["protocol_sha256"]:
            raise ValueError("Protocol changed since audit; run check-data --content-hash")
    return splits


def run_directory(cfg: dict, model: str, seed: int) -> Path:
    """Use explicit experiment output roots to keep ablations and seeds isolated."""
    return Path(cfg["output_root"]) / "runs" / model / f"seed_{seed}"


def provenance(cfg: dict, seed: int, model: str) -> dict:
    """Record configuration, runtime versions and official protocol hashes."""
    packages = ["torch", "torchvision", "numpy", "scipy", "librosa", "scikit-learn", "soundfile"]
    return {"config": cfg, "seed": seed, "model": model, "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {p: importlib.metadata.version(p) for p in packages},
            "protocol_sha256": {s: digest(protocol_path(Path(cfg["data_root"]), s)) for s in SPLITS}}


def fit_baseline(cfg: dict, name: str, seed: int) -> Path:
    """Fit SVM or boosting; choose predefined hyperparameter candidates by dev EER."""
    seed_all(seed, cfg["deep"]["threads"])
    splits = validated_splits(cfg)
    if cfg["augmentation"]["enabled"]:
        raise ValueError("Baseline augmentation is intentionally unsupported; primary comparison uses no augmentation")
    train, dev = splits["train"], splits["dev"]
    x_train, x_dev = (cache_features(records, cfg, "mfcc") for records in (train, dev))
    y_train, y_dev = (np.array([r.label for r in records]) for records in (train, dev))
    out = run_directory(cfg, name, seed)
    if (out / "run.json").exists() or (out / "eval").exists():
        raise FileExistsError(f"Existing run: {out}. Choose a new output_root to rerun")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.joblib").exists():
        (out / "model.joblib").replace(out / "model.interrupted.joblib")
    started = time.perf_counter()
    candidate_results = []
    best = None
    best_eer = float("inf")
    candidates = cfg["baseline"]["svm_c"] if name == "svm" else cfg["baseline"]["boosting_leaves"]
    with threadpool_limits(limits=cfg["deep"]["threads"]):
        for parameter in candidates:
            if name == "svm":
                model = make_pipeline(StandardScaler(), SVC(C=parameter, kernel="rbf", gamma="scale",
                                      class_weight="balanced", cache_size=512, random_state=seed))
            elif name == "boosting":
                model = HistGradientBoostingClassifier(max_leaf_nodes=parameter,
                    max_iter=cfg["baseline"]["boosting_iterations"], class_weight="balanced",
                    early_stopping=False, random_state=seed)
            else:
                raise ValueError(name)
            model.fit(x_train, y_train)
            scores = baseline_scores(model, x_dev)
            value = eer(y_dev, scores)
            candidate_results.append({"parameter": parameter, "dev_eer": value})
            print(f"{name} seed={seed}, parameter={parameter}, dev EER={value:.6f}", flush=True)
            if value < best_eer:
                best, best_eer = model, value
    assert best is not None
    scores = baseline_scores(best, x_dev)
    threshold = dev_threshold(y_dev, scores)
    joblib.dump(best, out / "model.joblib.tmp")
    (out / "model.joblib.tmp").replace(out / "model.joblib")
    metadata = provenance(cfg, seed, name)
    metadata.update(threshold=threshold, candidates=candidate_results,
                    training_seconds=time.perf_counter() - started,
                    model_bytes=(out / "model.joblib").stat().st_size)
    write_evaluation(dev, scores, threshold, out / "dev")
    save_json(out / "run.json", metadata)  # completion marker written last
    return out


def baseline_scores(model: object, features: np.ndarray) -> np.ndarray:
    """Get continuous bona fide decision values, never hard labels for EER."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    classes = list(model.classes_)
    return np.asarray(model.predict_proba(features)[:, classes.index(1)])


def fit_deep(cfg: dict, seed: int) -> Path:
    """Train weighted BCE MobileNetV2 with dev-EER checkpointing and early stopping."""
    seed_all(seed, cfg["deep"]["threads"])
    splits = validated_splits(cfg)
    train, dev = splits["train"], splits["dev"]
    out = run_directory(cfg, "mobile", seed)
    if (out / "run.json").exists() or (out / "eval").exists():
        raise FileExistsError(f"Existing run: {out}. Choose a new output_root to rerun")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        (out / "model.pt").replace(out / "model.interrupted.pt")
    train_cache = None if cfg["augmentation"]["enabled"] else cache_features(train, cfg, "spectrogram")
    dev_cache = cache_features(dev, cfg, "spectrogram")
    training = SpectrogramDataset(train, cfg, seed=seed, training=True, cached=train_cache)
    training_loader = loader(training, cfg, seed, shuffle=True)
    validation_loader = loader(SpectrogramDataset(dev, cfg, cached=dev_cache), cfg, seed)
    device = resolve_device(cfg)
    model = build_mobile().to(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    print(f"MobileNet training device: {device} ({device_name}); batch size={cfg['deep']['batch_size']}", flush=True)
    positives = sum(r.label for r in train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - positives) / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["deep"]["learning_rate"],
                                  weight_decay=cfg["deep"]["weight_decay"])
    best_eer, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(cfg["deep"]["epochs"]):
        epoch_started = time.perf_counter()
        training.epoch = epoch
        model.train()
        total_loss = 0.0
        for batch, (x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)
            loss = optimization_step(model, optimizer, criterion, x, y)
            total_loss += loss * len(y)
            if batch % 100 == 0:
                print(f"mobile seed={seed} epoch={epoch+1} batch={batch}/{len(training_loader)}", flush=True)
        scores = predict_deep(model, validation_loader, device)
        value = eer(np.array([r.label for r in dev]), scores)
        epoch_seconds = time.perf_counter() - epoch_started
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train), "dev_eer": value,
                        "epoch_seconds_including_dev": epoch_seconds})
        save_json(out / "history.json", history)
        print(f"mobile seed={seed} epoch={epoch+1} dev EER={value:.6f}; epoch+dev={epoch_seconds:.1f}s", flush=True)
        if value < best_eer:
            best_eer, stale = value, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, out / "model.pt.tmp")
            (out / "model.pt.tmp").replace(out / "model.pt")
        else:
            stale += 1
        if stale >= cfg["deep"]["patience"]:
            break
    if not (out / "model.pt").exists():
        raise RuntimeError("No checkpoint was trained; epochs must be positive")
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    scores = predict_deep(model, validation_loader, device)
    threshold = dev_threshold(np.array([r.label for r in dev]), scores)
    metadata = provenance(cfg, seed, "mobile")
    metadata.update(threshold=threshold, best_epoch=checkpoint["epoch"],
                    parameters=sum(p.numel() for p in model.parameters()), device=str(device), device_name=device_name,
                    training_seconds=time.perf_counter() - started,
                    model_bytes=(out / "model.pt").stat().st_size)
    write_evaluation(dev, scores, threshold, out / "dev")
    save_json(out / "run.json", metadata)  # completion marker written last
    return out


def evaluate_run(directory: Path, split: str = "eval") -> dict:
    """Evaluate a frozen run with its saved configuration and dev threshold."""
    import json
    if split == "eval":
        from .experiments import verify_frozen
        verify_frozen(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    cfg, seed, name = metadata["config"], metadata["seed"], metadata["model"]
    seed_all(seed, cfg["deep"]["threads"])
    splits = validated_splits(cfg)
    for s in SPLITS:
        if digest(protocol_path(Path(cfg["data_root"]), s)) != metadata["protocol_sha256"][s]:
            raise ValueError("Protocols differ from frozen training run")
    records = splits[split]
    started = time.perf_counter()
    if name == "mobile":
        device = resolve_device(cfg)
        model = build_mobile().to(device)
        checkpoint = torch.load(directory / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        cache = cache_features(records, cfg, "spectrogram")
        batches = loader(SpectrogramDataset(records, cfg, cached=cache), cfg, seed)
        scores = predict_deep(model, batches, device)
    else:
        model = joblib.load(directory / "model.joblib")
        with threadpool_limits(limits=cfg["deep"]["threads"]):
            scores = baseline_scores(model, cache_features(records, cfg, "mfcc"))
    seconds = time.perf_counter() - started
    result = write_evaluation(records, scores, metadata["threshold"], directory / split)
    save_json(directory / split / "timing.json", {"total_seconds_including_cache": seconds,
                                                 "trials": len(records)})
    return result
