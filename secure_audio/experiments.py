"""Predeclared ablations, experiment locking and seed aggregation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from .common import digest, read_config, save_json


def make_ablations(config_path: Path) -> list[Path]:
    """Write four one-factor contrasts against log-Mel/4s/no augmentation.

    Only the deep model changes representation/augmentation. Clip-length
    contrasts can also be applied to baselines using their generated config.
    """
    base = read_config(config_path)
    if (base["features"]["spectrogram"] != "logmel" or base["audio"]["seconds"] != 4
            or base["augmentation"]["enabled"]):
        raise ValueError("Ablation generator expects the default logmel/4s/no-augmentation reference")
    variants = [("mel", ("features", "spectrogram"), "mel"),
                ("length_2s", ("audio", "seconds"), 2.0),
                ("length_6s", ("audio", "seconds"), 6.0),
                ("augmentation", ("augmentation", "enabled"), True)]
    files = []
    for name, (section, field), value in variants:
        cfg = copy.deepcopy(base)
        cfg[section][field] = value
        cfg["output_root"] = str(Path(base["output_root"]) / "ablations" / name)
        path = Path("configs") / f"ablation_{name}.yaml"
        if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != cfg:
            raise FileExistsError(f"Refusing to overwrite modified ablation config: {path}")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        files.append(path)
    return files


def freeze_experiments(root: Path, expected_seeds: list[int]) -> dict:
    """Lock trained checkpoints and run metadata before any eval access.

    All runs present under root must have completed training. Eval refuses
    artifacts that differ from this manifest. This is an audit trail, not a
    cryptographic defense against an experimenter intentionally editing files.
    """
    files = sorted(root.glob("runs/*/seed_*/run.json"))
    if not files:
        raise ValueError(f"No completed runs under {root}")
    for directory in root.glob("runs/*/seed_*"):
        if directory.is_dir() and not (directory / "run.json").exists():
            raise ValueError(f"Incomplete run must finish before freezing: {directory}")
    manifest = {"expected_seeds": expected_seeds, "runs": {}}
    groups: dict[str, set[int]] = {}
    reference_config = None
    for run_file in files:
        meta = json.loads(run_file.read_text(encoding="utf-8"))
        if reference_config is None:
            reference_config = meta["config"]
        elif meta["config"] != reference_config:
            raise ValueError("Runs within one experiment root must use the same configuration")
        groups.setdefault(meta["model"], set()).add(meta["seed"])
        directory = run_file.parent
        if (directory / "eval").exists():
            raise ValueError(f"Eval already exists for {directory}; cannot claim a new pre-eval freeze")
        checkpoint = directory / ("model.pt" if meta["model"] == "mobile" else "model.joblib")
        manifest["runs"][str(directory.relative_to(root)).replace("\\", "/")] = {
            "run_sha256": digest(run_file), "checkpoint_sha256": digest(checkpoint)}
    for model, seeds in groups.items():
        if seeds != set(expected_seeds) or len(seeds) < 3:
            raise ValueError(f"{model}: need exactly {expected_seeds}, got {sorted(seeds)}")
    target = root / "frozen.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("Existing freeze differs; choose a fresh output root")
    else:
        save_json(target, manifest)
    return manifest


def verify_frozen(directory: Path) -> None:
    """Require unchanged checkpoint/config/threshold metadata before eval."""
    root = directory.parent.parent.parent
    freeze = root / "frozen.json"
    if not freeze.exists():
        raise ValueError(f"Freeze first: python -m secure_audio freeze --root \"{root}\"")
    manifest = json.loads(freeze.read_text(encoding="utf-8"))
    key = str(directory.relative_to(root)).replace("\\", "/")
    entry = manifest["runs"].get(key)
    if entry is None:
        raise ValueError("Run not part of frozen experiment")
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    checkpoint = directory / ("model.pt" if meta["model"] == "mobile" else "model.joblib")
    if entry != {"run_sha256": digest(directory / "run.json"), "checkpoint_sha256": digest(checkpoint)}:
        raise ValueError("Frozen artifact was modified")


def aggregate(root: Path, expected_seeds: list[int]) -> dict:
    """Aggregate per-run metrics (not pooled predictions); require >=3 seeds."""
    output = {}
    for model_dir in sorted((root / "runs").glob("*")):
        if not model_dir.is_dir():
            continue
        seed_dirs = sorted(model_dir.glob("seed_*"))
        found = [int(path.name.split("_")[1]) for path in seed_dirs]
        if set(found) != set(expected_seeds) or len(found) < 3:
            raise ValueError(f"Incomplete seed set for {model_dir}: {found}")
        output[model_dir.name] = {}
        for split in ("dev", "eval"):
            results = [json.loads((path / split / "metrics.json").read_text(encoding="utf-8")) for path in seed_dirs]
            combined = {}
            for metric in ("eer", "eer_interpolated_tie_aware", "accuracy", "balanced_accuracy", "roc_auc"):
                values = [r[metric] for r in results]
                combined[metric] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)),
                                    "min": float(np.min(values)), "max": float(np.max(values)), "values": values}
            combined["attacks"] = {}
            for attack in results[0]["attacks"]:
                if not all(r["attacks"][attack]["available"] for r in results):
                    combined["attacks"][attack] = {"available": False}
                    continue
                combined["attacks"][attack] = {"available": True}
                for metric in ("eer", "spoof_acceptance_rate"):
                    values = [r["attacks"][attack][metric] for r in results]
                    combined["attacks"][attack][metric] = {"mean": float(np.mean(values)),
                        "sample_std": float(np.std(values, ddof=1)), "values": values}
            output[model_dir.name][split] = combined
        probes = [path / "external/probe.json" for path in seed_dirs]
        if all(path.exists() for path in probes):
            values = [json.loads(path.read_text(encoding="utf-8"))["spoof_detection_rate_scored"] for path in probes]
            output[model_dir.name]["external_spoof_detection_rate"] = {
                "values": values, "mean": float(np.mean(values)) if None not in values else None,
                "sample_std": float(np.std(values, ddof=1)) if None not in values else None}
    if not output:
        raise ValueError(f"No runs found under {root}")
    save_json(root / "aggregate.json", output)
    return output
