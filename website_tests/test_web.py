"""Local website API tests; generated artifacts are never benchmark results."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pytest
import requests
import soundfile as sf
from sklearn.dummy import DummyClassifier

import web_server
from secure_audio.common import read_config


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Serve an isolated temporary artifact root on a random loopback port."""
    monkeypatch.setattr(web_server, "ROOT", tmp_path)
    web_server.DETECTORS.clear()
    instance = web_server.ThreadingHTTPServer(("127.0.0.1", 0), web_server.Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_port}", tmp_path
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)
    web_server.DETECTORS.clear()


def test_status_and_static_root(server) -> None:
    url, root = server
    response = requests.get(url + "/api/status", timeout=5)
    assert response.status_code == 200
    assert response.json()["service"] == "voxguard-local"
    assert response.json()["models"] == {"mobile": [], "svm": [], "boosting": []}
    assert requests.get(url, timeout=5).status_code == 200
    assert requests.get(url + "/../web_server.py", timeout=5).status_code == 404


def test_reject_untrained_model(server) -> None:
    url, _ = server
    response = requests.post(url + "/api/analyze?model=mobile&seed=13", data=b"wave", timeout=5)
    assert response.status_code == 409 and "not trained" in response.json()["error"]


@pytest.mark.parametrize("query", ["model=../../data&seed=13", "model=mobile&seed=99", "model=svm&seed=bad"])
def test_invalid_model_seed(server, query: str) -> None:
    url, _ = server
    assert requests.post(url + "/api/analyze?" + query, data=b"wave", timeout=5).status_code == 400


def test_origin_and_host_guards(server) -> None:
    url, _ = server
    assert requests.post(url + "/api/analyze?model=svm&seed=13", headers={"Origin": "https://untrusted.example"}, data=b"wave", timeout=5).status_code == 403
    assert requests.get(url + "/api/status", headers={"Host": "untrusted.example"}, timeout=5).status_code == 403


def test_empty_and_large_upload(server, monkeypatch: pytest.MonkeyPatch) -> None:
    url, _ = server
    assert requests.post(url + "/api/analyze?model=svm&seed=13", data=b"", timeout=5).status_code == 413
    monkeypatch.setattr(web_server, "MAX_UPLOAD", 3)
    assert requests.post(url + "/api/analyze?model=svm&seed=13", data=b"1234", timeout=5).status_code == 413


def test_detector_integration_and_cleanup(server, monkeypatch: pytest.MonkeyPatch) -> None:
    url, root = server
    run = root / "artifacts/runs/svm/seed_13"
    run.mkdir(parents=True)
    # A deliberately trivial test-only estimator tests HTTP -> loader -> score.
    estimator = DummyClassifier(strategy="prior").fit(np.zeros((4, 120)), [0, 0, 1, 1])
    joblib.dump(estimator, run / "model.joblib")
    cfg = read_config("configs/default.yaml")
    (run / "run.json").write_text(json.dumps({"config": cfg, "model": "svm", "seed": 13, "threshold": .6}), encoding="utf-8")
    assert requests.get(url + "/api/status", timeout=5).json()["models"]["svm"] == [{"seed": 13}]
    upload_dir = root / "temp_uploads"
    upload_dir.mkdir()
    original = web_server.tempfile.NamedTemporaryFile
    monkeypatch.setattr(web_server.tempfile, "NamedTemporaryFile", lambda **kwargs: original(dir=upload_dir, **kwargs))
    for silent in (False, True):
        wave = np.zeros(16000) if silent else .2 * np.sin(2*np.pi*300*np.arange(16000)/16000)
        body = io.BytesIO()
        sf.write(body, wave, 16000, format="WAV")
        response = requests.post(url + "/api/analyze?model=svm&seed=13", data=body.getvalue(), timeout=60)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == ("rejected" if silent else "scored")
        assert "file" not in result
        if not silent:
            assert result["bonafide_score"] == .5 and result["threshold_from_dev"] == .6
        assert not list(upload_dir.iterdir())
