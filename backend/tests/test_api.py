"""Basic API tests for Meri Awaaz backend."""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _make_silent_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    nframes = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


def test_health(client: TestClient):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["backend"] == "ok"
    assert "ffmpeg" in data
    assert "whisper" in data
    assert isinstance(data["whisper"], dict)
    assert "available" in data["whisper"]
    assert isinstance(data["tts"], dict)
    assert "supported_languages" in data["tts"]
    assert isinstance(data["gemini"], dict)
    assert "model" in data["gemini"]


def test_root(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "Meri Awaaz" in res.json()["name"]


def test_reject_invalid_extension(client: TestClient):
    res = client.post(
        "/api/upload",
        files={"file": ("malware.exe", b"not-audio", "application/octet-stream")},
    )
    assert res.status_code == 400
    body = res.json()
    assert body["success"] is False
    assert "Unsupported" in body["error"]


def test_reject_empty_file(client: TestClient):
    res = client.post(
        "/api/upload",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert res.status_code == 400
    assert res.json()["success"] is False


def test_upload_valid_wav(client: TestClient):
    wav = _make_silent_wav(1.0)
    res = client.post(
        "/api/upload",
        files={"file": ("sample.wav", wav, "audio/wav")},
    )
    # May fail if ffmpeg missing; otherwise should succeed
    if res.status_code == 503:
        assert "FFmpeg" in res.json()["error"]
        return
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "upload_id" in data

    # Process job creation
    proc = client.post(
        "/api/process",
        json={"upload_id": data["upload_id"], "output_language": "en"},
    )
    assert proc.status_code == 200
    job = proc.json()
    assert job["status"] == "processing"
    assert "job_id" in job

    status = client.get(f"/api/status/{job['job_id']}")
    assert status.status_code == 200
    assert status.json()["job_id"] == job["job_id"]


def test_process_missing_upload(client: TestClient):
    res = client.post(
        "/api/process",
        json={"upload_id": "does-not-exist", "output_language": "en"},
    )
    assert res.status_code == 404


def test_status_missing_job(client: TestClient):
    res = client.get("/api/status/missingjob")
    assert res.status_code == 404
