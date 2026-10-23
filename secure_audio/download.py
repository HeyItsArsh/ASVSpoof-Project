"""Resumable official LA archive acquisition with checksum and safe extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .common import save_json

API = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269"
LANDING = "https://datashare.ed.ac.uk/handle/10283/3336"


def segmented_download(part: Path, total: int, expected: str, connections: int) -> None:
    """Fetch bounded independent HTTP ranges; preserve an existing prefix.

    Per-range files permit safe retry after interruption. Assembly truncates to
    the original prefix before appending, so interrupted assembly is repeatable.
    Publisher checksum is still required before renaming/extracting the archive.
    """
    folder = part.parent / ".LA.segments"
    folder.mkdir(exist_ok=True)
    state_file = folder / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["total"] != total or state["checksum"] != expected:
            raise ValueError("Segment state does not match official archive metadata")
    else:
        state = {"prefix_bytes": part.stat().st_size if part.exists() else 0,
                 "total": total, "checksum": expected, "chunk_bytes": 16 * 1024**2}
        save_json(state_file, state)
    prefix = state["prefix_bytes"]
    if prefix > total or (prefix and (not part.exists() or part.stat().st_size < prefix)):
        raise ValueError("Download prefix is missing or invalid")
    ranges = [(start, min(start + state["chunk_bytes"], total) - 1)
              for start in range(prefix, total, state["chunk_bytes"])]

    def fetch(item: tuple[int, int]) -> int:
        """Resume a single exact range, returning its complete byte count."""
        start, end = item
        path = folder / f"{start:012d}.bin"
        length = end - start + 1
        for attempt in range(6):
            offset = path.stat().st_size if path.exists() else 0
            if offset == length:
                return length
            if offset > length:
                raise ValueError(f"Invalid range file size: {path}")
            try:
                with requests.get(API + "/content", stream=True, timeout=(30, 120),
                                  headers={"Range": f"bytes={start+offset}-{end}"}) as response:
                    response.raise_for_status()
                    wanted = f"bytes {start+offset}-{end}/{total}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != wanted:
                        raise RuntimeError("Server did not honor exact Range; rerun with --connections 1")
                    with path.open("ab") as out:
                        for block in response.iter_content(1024 * 1024):
                            out.write(block)
                if path.stat().st_size != length:
                    raise requests.ConnectionError("Incomplete range")
                return length
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 15))
        raise RuntimeError("Unreachable range retry state")

    completed = prefix
    last = time.monotonic()
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in as_completed(futures):
            completed += future.result()
            if time.monotonic() - last > 15 or completed == total:
                print(f"LA.zip ranges: {completed / total:.1%} ({completed / 1024**3:.2f} GiB)", flush=True)
                last = time.monotonic()
    with part.open("r+b" if part.exists() else "wb") as out:
        out.truncate(prefix)
        out.seek(prefix)
        for start, _ in ranges:
            with (folder / f"{start:012d}.bin").open("rb") as stream:
                shutil.copyfileobj(stream, out, 4 * 1024**2)


def checksum(path: Path, algorithm: str) -> str:
    """Stream an archive checksum using the publisher's algorithm."""
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    """Download LA.zip with HTTP Range resume and publisher checksum validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--connections", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(API, timeout=60)
    response.raise_for_status()
    meta = response.json()
    total = int(meta["sizeBytes"])
    expected = meta["checkSum"]["value"]
    algorithm = meta["checkSum"]["checkSumAlgorithm"].lower().replace("-", "")
    archive = args.directory / "LA.zip"
    part = args.directory / "LA.zip.part"
    if not archive.exists():
        offset = part.stat().st_size if part.exists() else 0
        reserve = (total - offset) + (20 * 1024**3 if args.extract else 0)
        if shutil.disk_usage(args.directory).free < reserve:
            raise RuntimeError(f"Need at least {reserve / 1024**3:.1f} GiB free for download/extraction")
        if args.connections > 1:
            segmented_download(part, total, expected, args.connections)
        for attempt in range(8 if args.connections == 1 else 0):
            offset = part.stat().st_size if part.exists() else 0
            if offset == total:
                break
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(API + "/content", headers=headers, stream=True, timeout=(30, 120)) as stream:
                    stream.raise_for_status()
                    if offset and stream.status_code != 206:
                        offset = 0  # server ignored Range; restart rather than append
                    if stream.status_code == 206 and not stream.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Server returned unexpected byte range")
                    last = time.monotonic()
                    with part.open("ab" if offset else "wb") as out:
                        for block in stream.iter_content(4 * 1024 * 1024):
                            if not block:
                                continue
                            out.write(block)
                            offset += len(block)
                            if time.monotonic() - last > 15:
                                print(f"LA.zip: {offset / total:.1%} ({offset / 1024**3:.2f} GiB)", flush=True)
                                last = time.monotonic()
                if offset != total:
                    raise requests.ConnectionError(f"Incomplete transfer {offset}/{total}")
                break
            except requests.RequestException as exc:
                if attempt == 7:
                    raise
                print(f"Transfer interrupted; retrying from disk: {exc}", flush=True)
                time.sleep(min(2**attempt, 30))
        if part.stat().st_size != total or checksum(part, algorithm) != expected:
            raise RuntimeError("Archive size/checksum mismatch; keep .part for diagnosis, do not extract")
        part.replace(archive)
    if archive.stat().st_size != total or checksum(archive, algorithm) != expected:
        raise RuntimeError("Existing archive does not match publisher checksum")
    save_json(args.directory / "download_receipt.json", {
        "source": LANDING, "api": API, "bytes": total, "checksum": expected,
        "algorithm": algorithm, "verified": True})
    segments = args.directory / ".LA.segments"
    if segments.exists():
        for segment in segments.glob("*.bin"):
            segment.unlink()
        (segments / "state.json").unlink(missing_ok=True)
        if not any(segments.iterdir()):
            segments.rmdir()
    if args.extract:
        with zipfile.ZipFile(archive) as bundle:
            uncompressed = sum(member.file_size for member in bundle.infolist())
            if shutil.disk_usage(args.directory).free < uncompressed:
                raise RuntimeError("Insufficient space to extract archive")
            for member in bundle.infolist():
                target = (args.directory / member.filename).resolve()
                if not target.is_relative_to(args.directory.resolve()):
                    raise RuntimeError(f"Unsafe archive path: {member.filename}")
                bundle.extract(member, args.directory)
        print("Extracted. Run: python -m secure_audio check-data --content-hash", flush=True)


if __name__ == "__main__":
    main()
