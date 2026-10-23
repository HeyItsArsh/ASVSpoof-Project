"""Serve the VoxGuard website and real local inference: python web_server.py.

Bound exclusively to loopback. No new third-party web framework is required.
The hosted static site is a presentation; local Python supplies real models.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "website/dist"
MODELS = ("svm", "boosting", "mobile")
SEEDS = (13, 37, 73)
MAX_UPLOAD = 25 * 1024 * 1024
INFERENCE_LOCK = threading.Lock()
DETECTORS: dict[tuple[str, int, int, int], Any] = {}


def read_json(path: Path) -> dict:
    """Read completed JSON atomically, tolerating absent or in-progress artifacts."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def available_models() -> dict[str, list[dict]]:
    """Expose only completed primary runs with both metadata and model files."""
    result: dict[str, list[dict]] = {model: [] for model in MODELS}
    for model in MODELS:
        for seed in SEEDS:
            run = ROOT / f"artifacts/runs/{model}/seed_{seed}"
            checkpoint = run / ("model.pt" if model == "mobile" else "model.joblib")
            metadata = read_json(run / "run.json")
            if checkpoint.is_file() and metadata.get("model") == model and metadata.get("seed") == seed:
                result[model].append({"seed": seed})
    return result


class Handler(SimpleHTTPRequestHandler):
    """Same-origin, bounded-upload API and a static-only document root."""
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Keep static requests inside website/dist, away from data/checkpoints."""
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def trusted_request(self, *, mutation: bool = False) -> bool:
        """Reject DNS rebinding and cross-origin uploads; permit local CLI tests."""
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed:
            self.json_response(403, {"error": "Only localhost requests are accepted."})
            return False
        origin = self.headers.get("Origin")
        if mutation and origin and origin != f"http://{host}":
            self.json_response(403, {"error": "Cross-origin uploads are not allowed."})
            return False
        return True

    def end_headers(self) -> None:
        """Send explicit no-sniff and privacy headers without enabling CORS."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        super().end_headers()

    def json_response(self, status: int, payload: dict) -> None:
        """Serialize finite JSON with a precise body length."""
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Expose live status/aggregates, otherwise serve only static assets."""
        if not self.trusted_request():
            return
        if urlsplit(self.path).path == "/api/status":
            pipeline = read_json(ROOT / "artifacts/pipeline/status.json")
            # No local paths, command arguments or uploaded audio are disclosed.
            state = {k: pipeline[k] for k in ("state", "current_step", "completed") if k in pipeline}
            self.json_response(200, {"service": "voxguard-local", "models": available_models(),
                                      "pipeline": state, "aggregate": read_json(ROOT / "artifacts/aggregate.json")})
        elif urlsplit(self.path).path.startswith("/api/"):
            self.json_response(404, {"error": "Unknown API route."})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        """Classify one bounded audio body; clean up temporary files in all cases."""
        self.close_connection = True  # rejected/unread uploads never poison keep-alive
        if not self.trusted_request(mutation=True):
            return
        parsed = urlsplit(self.path)
        if parsed.path != "/api/analyze":
            self.json_response(404, {"error": "Unknown API route."})
            return
        query = parse_qs(parsed.query)
        model = query.get("model", [""])[0]
        try:
            seed = int(query.get("seed", [""])[0])
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.json_response(400, {"error": "Invalid seed or body length."})
            return
        if model not in MODELS or seed not in SEEDS:
            self.json_response(400, {"error": "Choose a supported model and seed."})
            return
        if length <= 0 or length > MAX_UPLOAD:
            self.json_response(413, {"error": "Choose a nonempty audio file up to 25 MB."})
            return
        if not any(row["seed"] == seed for row in available_models()[model]):
            self.json_response(409, {"error": "This model run is not trained yet. Let training finish, then refresh."})
            return
        if not INFERENCE_LOCK.acquire(blocking=False):
            self.json_response(429, {"error": "Another recording is being analyzed. Try again shortly."})
            return
        temporary: Path | None = None
        try:
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                self.json_response(400, {"error": "Incomplete audio upload."})
                return
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as out:
                temporary = Path(out.name)
                out.write(body)
            from secure_audio.inference import Detector
            run = ROOT / f"artifacts/runs/{model}/seed_{seed}"
            weights = run / ("model.pt" if model == "mobile" else "model.joblib")
            key = (model, seed, weights.stat().st_mtime_ns, (run / "run.json").stat().st_mtime_ns)
            if key not in DETECTORS:
                # Bound memory: retain one loaded model on a normal laptop.
                DETECTORS.clear()
                DETECTORS[key] = Detector(run)
            result = DETECTORS[key].score(temporary)
            result.pop("file", None)
            # Finish privacy cleanup before acknowledging a successful request.
            temporary.unlink()
            temporary = None
            self.json_response(200, result)
        except (TimeoutError, OSError, ValueError, RuntimeError):
            self.json_response(422, {"error": "Audio analysis failed. Try a valid WAV/FLAC recording and check your local model files."})
        finally:
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            finally:
                INFERENCE_LOCK.release()


def main() -> None:
    """Launch the local application without starting or changing model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(ROOT)  # saved experiment configs use repository-relative paths
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"VoxGuard website: http://127.0.0.1:{server.server_port}", flush=True)
    print("Press Ctrl+C to stop. Training remains a separate process.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
