"""Offline HTTP range/assembly tests: no network or real archive needed."""
from pathlib import Path

import pytest

from secure_audio import download


class Response:
    def __init__(self, data: bytes, start: int, end: int, *, honor: bool = True) -> None:
        self.data, self.start, self.end = data, start, end
        self.status_code = 206 if honor else 200
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, size: int):
        yield self.data[self.start:self.end + 1]


def test_segment_download_preserves_prefix_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"official archive fixture" * 100
    part = tmp_path / "LA.zip.part"
    part.write_bytes(data[:37])
    calls = []

    def get(url, **kwargs):
        bounds = kwargs["headers"]["Range"].removeprefix("bytes=").split("-")
        start, end = map(int, bounds)
        calls.append((start, end))
        return Response(data, start, end)

    monkeypatch.setattr(download.requests, "get", get)
    download.segmented_download(part, len(data), "checksum", 2)
    assert part.read_bytes() == data
    assert calls == [(37, len(data)-1)]
    # An interrupted assembly is reassembled from the original prefix and ranges.
    part.write_bytes(data[:100])
    download.segmented_download(part, len(data), "checksum", 2)
    assert part.read_bytes() == data and len(calls) == 1


def test_range_ignored_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"example"
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: Response(data, 0, len(data)-1, honor=False))
    with pytest.raises(RuntimeError, match="Range"):
        download.segmented_download(tmp_path / "LA.zip.part", len(data), "checksum", 2)
