"""Run the full declared study sequentially; retain logs/status and allow resume.

Usage: .venv/Scripts/python.exe scripts/run_all.py [--resume]
No subprocess is detached. Interrupting the runner stops its current child.
Large downloads and full CPU experiments may take many hours or days.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from secure_audio.common import save_json


@contextmanager
def pipeline_lock(path: Path) -> Iterator[None]:
    """Hold an OS file lock; crashes release it and a second runner fails fast."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit("Another pipeline is already running. Inspect artifacts/pipeline/status.json.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main() -> None:
    """Execute explicit steps with durable status; stop immediately on failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = REPO / "artifacts" / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    status_file = directory / "status.json"
    code_hash = hashlib.sha256()
    inputs = [*sorted((REPO / "secure_audio").glob("*.py")),
              *sorted((REPO / "tests").glob("*.py")), REPO / "configs/default.yaml",
              REPO / "requirements.txt", REPO / "requirements-lock.txt", Path(__file__)]
    for path in inputs:
        code_hash.update(path.name.encode())
        code_hash.update(path.read_bytes())
    fingerprint = code_hash.hexdigest()
    status = {"study_fingerprint": fingerprint, "completed": [], "state": "starting"}
    if status_file.exists():
        previous = json.loads(status_file.read_text(encoding="utf-8"))
        if not args.resume:
            raise SystemExit("Existing pipeline status; use --resume to continue")
        if previous["study_fingerprint"] != fingerprint:
            raise SystemExit("Source/config changed since pipeline start. Inspect prior runs before restarting")
        status = previous
    py = sys.executable
    app = [py, "-m", "secure_audio"]
    steps = [
        ("unit_tests", [py, "-m", "pytest", "-q"]),
        ("synthetic_sanity", app + ["sanity"]),
        ("download", [py, "-m", "secure_audio.download", "--extract"]),
        ("audit", app + ["check-data", "--content-hash"]),
        ("real_sanity", app + ["sanity", "--real-data"]),
        ("ablation_configs", app + ["ablations"]),
        ("train_primary", app + ["train", "--resume"]),
    ]
    variants = ["mel", "length_2s", "length_6s", "augmentation"]
    for variant in variants:
        steps.append((f"train_{variant}", app + ["--config", f"configs/ablation_{variant}.yaml", "train", "--model", "mobile", "--resume"]))
    configs = [("primary", "configs/default.yaml")] + [(v, f"configs/ablation_{v}.yaml") for v in variants]
    for name, config in configs:
        steps.append((f"freeze_{name}", app + ["--config", config, "freeze"]))
    for name, config in configs:
        steps.extend([(f"evaluate_{name}", app + ["--config", config, "evaluate"]),
                      (f"aggregate_{name}", app + ["--config", config, "aggregate"])])
    # Local synthesis already present is preserved; otherwise the generation step is explicit.
    if not (REPO / "data/synthetic/manifest.json").exists():
        steps.append(("local_tts", ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/generate_tts.ps1"]))
    for model in ("svm", "boosting", "mobile"):
        for seed in (13, 37, 73):
            steps.append((f"external_{model}_{seed}", app + ["external", "--run", f"artifacts/runs/{model}/seed_{seed}"]))
    steps.append(("aggregate_with_external", app + ["aggregate"]))
    for name, command in steps:
        if name in status["completed"]:
            continue
        status.update(state="running", current_step=name, command=command,
                      updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(status_file, status)
        print(f"Running {name}: {' '.join(command)}", flush=True)
        with (directory / f"{name}.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                status.update(state="interrupted", current_step=name)
                save_json(status_file, status)
                raise
        if code:
            status.update(state="failed", return_code=code, updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(status_file, status)
            raise SystemExit(f"{name} failed ({code}); inspect {directory / (name + '.log')}")
        status["completed"].append(name)
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(status_file, status)
    status.update(state="complete", current_step=None, updated_utc=datetime.now(timezone.utc).isoformat())
    save_json(status_file, status)
    print("Study complete; see artifacts/aggregate.json and ablation aggregate files.")


if __name__ == "__main__":
    with pipeline_lock(REPO / "artifacts/pipeline/runner.lock"):
        main()
