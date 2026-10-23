"""Countermeasure metrics; larger scores always mean more bona fide."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from .common import save_json
from .data import Record


def det_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ASVspoof reference DET convention: stable-sort bona fide then spoof scores.

    At each sorted score reject all observations up to that position. This matches
    the legacy reference implementation (including its ordering of exact ties).
    Tied score counts are reported because partial tie rejection is not a feasible
    deployment threshold. Operational thresholds instead use grouped ROC points.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Expected aligned finite one-dimensional labels and scores")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("EER requires both bona fide (1) and spoof (0)")
    bona, spoof = scores[labels == 1], scores[labels == 0]
    combined = np.concatenate([bona, spoof])
    keys = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(combined, kind="mergesort")
    keys = keys[order]
    bona_rejected = np.cumsum(keys)
    spoof_accepted = len(spoof) - (np.arange(1, len(keys) + 1) - bona_rejected)
    frr = np.concatenate([[0.0], bona_rejected / len(bona)])
    far = np.concatenate([[1.0], spoof_accepted / len(spoof)])
    thresholds = np.concatenate([[np.nextafter(combined.min(), -np.inf)], combined[order]])
    return frr, far, thresholds


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return reference discrete EER as a fraction; no eval threshold is deployed."""
    frr, far, _ = det_curve(labels, scores)
    index = int(np.argmin(np.abs(frr - far)))
    return float((frr[index] + far[index]) / 2)


def dev_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select the attainable dev threshold minimizing |FRR-FAR|; predict score>=t."""
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    index = int(np.argmin(np.abs((1 - tpr) - fpr)))
    threshold = thresholds[index]
    if not np.isfinite(threshold):
        threshold = np.nextafter(float(np.max(scores)), np.inf)
    return float(threshold)


def summary(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Report rank metrics and fixed-threshold classification metrics."""
    det_curve(labels, scores)  # validate even if callers bypass eer
    predicted = np.asarray(scores) >= threshold
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    # A tie-aware linearly interpolated EER is supplementary, not the reference EER.
    difference = 1 - tpr - fpr
    crossing = int(np.where(difference <= 0)[0][0])
    if crossing == 0:
        interpolated = float(fpr[0])
    else:
        weight = difference[crossing - 1] / (difference[crossing - 1] - difference[crossing])
        interpolated = float(fpr[crossing - 1] + weight * (fpr[crossing] - fpr[crossing - 1]))
    return {"eer": eer(labels, scores), "eer_interpolated_tie_aware": interpolated,
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "roc_auc": float(roc_auc_score(labels, scores)), "threshold_from_dev": threshold,
            "confusion_matrix_labels": ["spoof", "bonafide"],
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            "tied_observations": int(len(scores) - len(np.unique(scores)))}


def write_evaluation(records: list[Record], scores: np.ndarray, threshold: float, out: Path) -> dict:
    """Save aligned scores, full metrics, per-attack diagnostics and plot images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    labels = np.array([r.label for r in records])
    result = summary(labels, scores, threshold)
    attacks = np.array([r.attack for r in records])
    result["attacks"] = {}
    for index in range(1, 20):
        attack = f"A{index:02d}"
        spoof_mask = attacks == attack
        if not spoof_mask.any():
            result["attacks"][attack] = {"available": False}
            continue
        mask = (labels == 1) | spoof_mask
        attack_result = summary(labels[mask], scores[mask], threshold)
        attack_result.update(available=True, n_spoof=int(spoof_mask.sum()),
                             spoof_acceptance_rate=float(np.mean(scores[spoof_mask] >= threshold)))
        result["attacks"][attack] = attack_result
    save_json(out / "metrics.json", result)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utterance", "speaker", "attack", "label", "bonafide_score"])
        writer.writerows((r.utterance, r.speaker, r.attack, r.label, float(s)) for r, s in zip(records, scores, strict=True))
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    np.savez_compressed(out / "roc.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={result['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="Spoof acceptance rate", ylabel="Bona fide acceptance rate", title="ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=160)
    plt.close(fig)
    cm = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["spoof", "bonafide"],
           yticklabels=["spoof", "bonafide"], xlabel="Predicted", ylabel="True", title="Dev-selected threshold")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return result
