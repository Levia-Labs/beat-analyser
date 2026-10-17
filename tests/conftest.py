"""Shared fixtures. The app is tested without beatcore or the note model: analysis.analyse is
replaced by a stub that writes one CSV, so the HTTP and storage behaviour runs anywhere Flask
and numpy do."""

import io
import os
import wave

import pytest
from hypothesis import settings

import analysis
import app as app_module

# Fast in task test and the Stop hook; task test:deep sets HYPOTHESIS_PROFILE=deep.
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("deep", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def wav_bytes(seconds: float = 0.1, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def upload_root(tmp_path):
    return tmp_path / "uploads"


@pytest.fixture
def client(upload_root, monkeypatch):
    def fake_analyse(audio_path, folder):
        analysis.write_tracks(folder, {"kick": [(0.5, "kick")]})
        return {"bpm": 128.0, "confidence": 0.5, "key": "A minor"}

    monkeypatch.setattr(analysis, "analyse", fake_analyse)
    app_module.app.config.update(TESTING=True, UPLOAD_ROOT=upload_root)
    return app_module.app.test_client()
