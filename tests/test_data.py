from dataclasses import replace
from pathlib import Path

import pytest

from secure_audio.data import SPLITS, Record, assert_disjoint, audit, protocol_path, read_protocol


def test_disjoint_and_protocol_shapes(tiny_dataset: dict) -> None:
    root = Path(tiny_dataset["data_root"])
    splits = {s: read_protocol(root, s) for s in SPLITS}
    assert_disjoint(splits)
    assert all(len(records) == 8 for records in splits.values())
    assert audit(root, content_hash=True)["decoded_content_checked"]


@pytest.mark.parametrize("field", ["speaker", "utterance", "path"])
def test_overlap_rejected(field: str) -> None:
    a = Record("speaker1", "LA_T_1", "-", 1, Path("train/a.flac"))
    b = Record("speaker2", "LA_D_1", "-", 1, Path("dev/b.flac"))
    b = replace(b, **{field: getattr(a, field)})
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"train": [a], "dev": [b]})


def test_duplicate_content_rejected(tiny_dataset: dict) -> None:
    root = Path(tiny_dataset["data_root"])
    a = read_protocol(root, "train")[0]
    b = read_protocol(root, "dev")[0]
    b.path.write_bytes(a.path.read_bytes())
    with pytest.raises(ValueError, match="Decoded audio leakage"):
        audit(root, content_hash=True)


def test_missing_file_rejected(tiny_dataset: dict) -> None:
    root = Path(tiny_dataset["data_root"])
    read_protocol(root, "train")[0].path.unlink()
    with pytest.raises(ValueError, match="missing"):
        audit(root)


def test_official_asv_enrollment_is_excluded(tiny_dataset: dict) -> None:
    root = Path(tiny_dataset["data_root"])
    record = read_protocol(root, "dev")[0]
    extra = record.path.with_name("LA_D_A1234567.flac")
    extra.write_bytes(record.path.read_bytes())
    with pytest.raises(ValueError, match="extra"):
        audit(root)
    protocol = root / "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("speaker_dev_0 LA_D_A1234567\n", encoding="utf-8")
    assert audit(root, content_hash=True)["dev"]["asv_enrollment_files_excluded"] == 1


def test_duplicate_protocol_trial_rejected(tiny_dataset: dict) -> None:
    path = protocol_path(Path(tiny_dataset["data_root"]), "train")
    text = path.read_text()
    path.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_protocol(Path(tiny_dataset["data_root"]), "train")


def test_official_dataset_when_present() -> None:
    root = Path("data/LA")
    if not protocol_path(root, "train").exists():
        pytest.skip("Official LA archive not downloaded/extracted yet")
    splits = {s: read_protocol(root, s) for s in SPLITS}
    assert_disjoint(splits)
