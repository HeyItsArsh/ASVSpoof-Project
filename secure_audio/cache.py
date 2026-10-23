"""Disk-backed feature caches with protocol/config/source-metadata identities."""
from __future__ import annotations

import hashlib
import json
import shutil
import os
from pathlib import Path

import numpy as np

from .audio import fix_length, load_audio, mel_spectrogram, mfcc_statistics
from .common import save_json
from .data import Record


def cache_features(records: list[Record], cfg: dict, kind: str) -> np.ndarray:
    """Build an atomic float32 NPY memmap, then reopen read-only.

    Bounded RAM even for full spectrogram arrays. Source file size/mtime and
    resolved path invalidate a stale cache. Run content-hash audit independently
    to verify cross-split content; cache identities are not a substitute for it.
    """
    if kind not in {"mfcc", "spectrogram"}:
        raise ValueError(kind)
    metadata = {}
    for folder in {r.path.parent for r in records}:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name.endswith(".flac"):
                    metadata[folder / entry.name] = entry.stat()
    identity = {"version": 2, "kind": kind, "audio": cfg["audio"], "features": cfg["features"],
                "records": [(r.utterance, str(r.path.absolute()), metadata[r.path].st_size,
                              metadata[r.path].st_mtime_ns) for r in records]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(cfg.get("cache_root", str(Path(cfg["output_root"]) / "cache")))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{kind}_{key}.npy"
    if target.exists():
        return np.load(target, mmap_mode="r")
    fn = mfcc_statistics if kind == "mfcc" else mel_spectrogram
    first = fn(fix_length(load_audio(records[0].path, cfg["audio"]), cfg["audio"]), cfg)
    temporary = target.with_suffix(".partial.npy")
    needed = len(records) * first.size * np.dtype(np.float32).itemsize
    if shutil.disk_usage(directory).free < needed + 1024**3:
        raise RuntimeError(f"Feature cache needs {needed / 1024**3:.2f} GiB plus 1 GiB free reserve")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                     shape=(len(records), *first.shape))
    array[0] = first
    for index, record in enumerate(records[1:], 1):
        wave = fix_length(load_audio(record.path, cfg["audio"]), cfg["audio"])
        array[index] = fn(wave, cfg)
        if index % 1000 == 0:
            print(f"Cache {kind}: {index}/{len(records)}", flush=True)
    array.flush()
    del array
    temporary.replace(target)
    save_json(target.with_suffix(".json"), {"cache_key": key, "kind": kind,
                                          "n_trials": len(records), "audio": cfg["audio"],
                                          "features": cfg["features"]})
    return np.load(target, mmap_mode="r")
