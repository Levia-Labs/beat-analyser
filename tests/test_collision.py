"""A name collision compares audio hashes (ADR 0014): same audio is the cached ZIP, different
audio is analysed again."""

import hashlib
import io
import json
import zipfile

import pytest
from test_app import upload

import analysis
import pipeline
from conftest import wav_bytes


@pytest.fixture
def calls(monkeypatch):
    runs = []

    def fake_analyse(audio_path, folder, progress=None):
        runs.append(audio_path.read_bytes())
        analysis.write_tracks(folder, {f"run{len(runs)}": [(0.5, "kick")]})
        return {"bpm": 128.0, "confidence": 0.5, "key": "A minor"}

    monkeypatch.setattr(analysis, "analyse", fake_analyse)
    return runs


def names(response):
    with zipfile.ZipFile(io.BytesIO(response.data)) as z:
        return set(z.namelist())


def test_same_file_twice_analyses_once(client, upload_root, calls):
    assert upload(client).status_code == 200
    assert upload(client).status_code == 200
    assert len(calls) == 1
    meta = json.loads((upload_root / "song" / "meta.json").read_text(encoding="utf-8"))
    assert meta["audio_sha256"] == hashlib.sha256(wav_bytes()).hexdigest()
    assert not list((upload_root / "song").glob("*.new*"))


def test_different_audio_same_name_analyses_again(client, upload_root, calls):
    first, second = wav_bytes(0.1), wav_bytes(0.2)
    upload(client, data=first)
    response = upload(client, data=second)
    assert response.status_code == 200
    assert calls == [first, second]
    assert names(response) == {"song/audio.wav", "song/meta.json", "song/run2.csv"}
    assert not (upload_root / "song" / "run1.csv").exists()
    meta = json.loads((upload_root / "song" / "meta.json").read_text(encoding="utf-8"))
    assert meta["audio_sha256"] == hashlib.sha256(second).hexdigest()


def test_meta_from_before_the_hash_matches_by_audio(client, upload_root, calls):
    upload(client)
    meta_path = upload_root / "song" / "meta.json"
    meta_path.write_text(
        json.dumps({"name": "song", "key": "A minor", "bpm": 128.0}), encoding="utf-8"
    )
    assert upload(client).status_code == 200
    assert len(calls) == 1
    upload(client, data=wav_bytes(0.3))
    assert len(calls) == 2


def test_failure_while_placing_keeps_the_old_project(client, upload_root, calls, monkeypatch):
    upload(client)
    before = (upload_root / "song_tracks.zip").read_bytes()

    def broken(wav):
        wav.write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        pipeline.build(upload_root, "song", broken)
    assert (upload_root / "song_tracks.zip").read_bytes() == before
    assert (upload_root / "song" / "audio.wav").read_bytes() == wav_bytes()
    assert (upload_root / "song" / "run1.csv").is_file()
    assert sorted(p.name for p in (upload_root / "song").iterdir()) == [
        "audio.wav",
        "meta.json",
        "run1.csv",
    ]
