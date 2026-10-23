"""Strict parsing and auditing of the official LA countermeasure protocols."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
from itertools import combinations

import numpy as np
import soundfile as sf

from .common import digest

SPLITS = ("train", "dev", "eval")
COUNTS = {"train": (2580, 22800), "dev": (2548, 22296), "eval": (7355, 63882)}


@dataclass(frozen=True)
class Record:
    """A single trial; label 1 denotes bona fide, label 0 denotes spoof."""
    speaker: str
    utterance: str
    attack: str
    label: int
    path: Path


def protocol_path(root: Path, split: str) -> Path:
    """Locate precisely the official CM protocol, never an ASV protocol."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    suffix = "trn" if split == "train" else "trl"
    return root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{suffix}.txt"


def read_protocol(root: Path, split: str) -> list[Record]:
    """Parse five fields, validate labels, attack IDs and utterance uniqueness."""
    records: list[Record] = []
    seen: set[str] = set()
    allowed = {f"A{i:02d}" for i in (range(1, 7) if split != "eval" else range(7, 20))}
    for number, line in enumerate(protocol_path(root, split).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{split} line {number}: expected 5 fields")
        speaker, uid, placeholder, attack, key = fields
        if placeholder != "-" or key not in {"bonafide", "spoof"}:
            raise ValueError(f"{split} line {number}: invalid CM fields")
        if not uid.startswith({"train": "LA_T_", "dev": "LA_D_", "eval": "LA_E_"}[split]):
            raise ValueError(f"Invalid trial ID: {uid}")
        if uid in seen or Path(uid).name != uid:
            raise ValueError(f"Duplicate or unsafe trial ID: {uid}")
        if (key == "bonafide" and attack != "-") or (key == "spoof" and attack not in allowed):
            raise ValueError(f"Invalid attack/key combination: {attack}/{key}")
        seen.add(uid)
        records.append(Record(speaker, uid, attack, int(key == "bonafide"),
                              root / f"ASVspoof2019_LA_{split}" / "flac" / f"{uid}.flac"))
    if not records:
        raise ValueError(f"Empty {split} protocol")
    return records


def assert_disjoint(splits: dict[str, list[Record]]) -> None:
    """Reject any cross-split overlap in speaker, trial ID or resolved file path."""
    # Resolve each directory once. Per-file resolve() on a cloud-backed Windows
    # folder is extremely slow; scandir identifies real file symlinks cheaply.
    folders: dict[Path, tuple[Path, set[str]]] = {}
    signatures = {}
    for split, records in splits.items():
        paths = set()
        for record in records:
            parent = record.path.parent
            if parent not in folders:
                try:
                    with os.scandir(parent) as entries:
                        links = {entry.name for entry in entries if entry.is_symlink()}
                except FileNotFoundError:
                    links = set()
                folders[parent] = (parent.resolve(), links)
            resolved_parent, links = folders[parent]
            resolved = record.path.resolve() if record.path.name in links else resolved_parent / record.path.name
            paths.add(os.path.normcase(str(resolved)))
        signatures[split] = {"speaker": {r.speaker for r in records},
                             "utterance": {r.utterance for r in records}, "path": paths}
    for left, right in combinations(splits, 2):
        for name in ("speaker", "utterance", "path"):
            overlap = signatures[left][name] & signatures[right][name]
            if overlap:
                raise ValueError(f"{name} leakage between {left}/{right}: {list(overlap)[:5]}")


def audit(root: Path, *, content_hash: bool = False) -> dict:
    """Validate exact official trial counts, files, sample rates and split isolation.

    Optional decoded PCM hashes detect copied content under different filenames.
    Hashes ignore FLAC metadata and are compared across splits, not within a split.
    """
    splits = {s: read_protocol(root, s) for s in SPLITS}
    assert_disjoint(splits)
    result = {}
    content_seen: dict[str, str] = {}
    for split, records in splits.items():
        counts = tuple(sum(r.label == label for r in records) for label in (1, 0))
        if counts != COUNTS[split]:
            raise ValueError(f"{split}: counts {counts}, expected {COUNTS[split]}")
        expected = {r.path.name for r in records}
        actual = {p.name for p in records[0].path.parent.glob("*.flac")}
        # Dev/eval also ship ASV enrollment audio, outside the CM trial list.
        # Allow only IDs explicitly listed in the official enrollment protocols.
        enrollment: set[str] = set()
        for gender in ("male", "female"):
            path = root / "ASVspoof2019_LA_asv_protocols" / f"ASVspoof2019.LA.asv.{split}.{gender}.trn.txt"
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 2:
                        raise ValueError(f"Invalid ASV enrollment row in {path}")
                    for uid in fields[1].split(","):
                        if Path(uid).name != uid or not uid.startswith("LA_"):
                            raise ValueError(f"Invalid ASV enrollment ID: {uid}")
                        enrollment.add(f"{uid}.flac")
        unexpected = actual - expected - enrollment
        if expected - actual or unexpected:
            raise ValueError(f"{split}: missing={list(expected-actual)[:5]}, extra={list(unexpected)[:5]}")
        for index, record in enumerate(records, 1):
            info = sf.info(record.path)
            if info.samplerate != 16000 or info.channels != 1 or info.frames == 0:
                raise ValueError(f"Unexpected audio format: {record.path}: {info}")
            if content_hash:
                pcm, _ = sf.read(record.path, dtype="int32")
                key = hashlib.sha256(np.asarray(pcm, dtype="<i4").tobytes()).hexdigest()
                previous = content_seen.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"Decoded audio leakage: {record.path} also in {previous}")
            if index % 1000 == 0 or index == len(records):
                print(f"Audit {split}: {index}/{len(records)} trials checked", flush=True)
        result[split] = {"bonafide": counts[0], "spoof": counts[1],
                         "asv_enrollment_files_excluded": len(actual & enrollment),
                         "speakers": len({r.speaker for r in records}),
                         "protocol_sha256": digest(protocol_path(root, split))}
    result["decoded_content_checked"] = content_hash
    return result
