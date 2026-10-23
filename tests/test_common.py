"""Regression tests for concurrent JSON writes and temporary Windows locks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from secure_audio.common import save_json


def test_concurrent_json_writers(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: save_json(path, {"writer": index, "data": "x" * 1000}), range(40)))
    result = json.loads(path.read_text())
    assert result["writer"] in range(40) and result["data"] == "x" * 1000
    assert not list(tmp_path.glob("*.tmp"))


def test_json_retries_transient_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.replace
    attempts = []

    def locked(self: Path, target: Path) -> Path:
        attempts.append(self)
        if len(attempts) < 3:
            raise PermissionError("Simulated Windows sharing violation")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", locked)
    save_json(tmp_path / "status.json", {"state": "running"})
    assert len(attempts) == 3
    assert json.loads((tmp_path / "status.json").read_text()) == {"state": "running"}


def test_pipeline_lock_excludes_duplicate_runner(tmp_path: Path) -> None:
    from scripts.run_all import pipeline_lock
    path = tmp_path / "runner.lock"
    with pipeline_lock(path):
        with pytest.raises(SystemExit, match="already running"):
            with pipeline_lock(path):
                pytest.fail("Second pipeline acquired the same lock")
    with pipeline_lock(path):
        pass
