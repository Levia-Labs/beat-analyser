"""Analysis progress the SPA polls while its POST is open."""

import io

import pytest

import analysis
import progress
from conftest import wav_bytes

PID = "0123456789abcdef0123456789abcdef"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def empty_registry():
    progress._entries.clear()
    yield
    progress._entries.clear()


def current(clock=None) -> dict:
    state = progress.snapshot(PID, clock=clock) if clock else progress.snapshot(PID)
    assert state is not None
    return state


def test_a_tracker_walks_its_stages():
    clock = Clock()
    tracker = progress.Tracker(PID, progress.YOUTUBE_STAGES, clock=clock)
    tracker.fraction(0.25)
    assert current(clock)["fraction"] == 0.25
    tracker.stage("rhythm")
    clock.now = 103.0
    state = current(clock)
    assert state["stage"] == "rhythm"
    assert state["done"] == 2
    assert state["total"] == 6
    assert state["fraction"] is None
    assert state["elapsed"] == 3.0
    tracker.finish()
    state = current(clock)
    assert state["finished"] is True
    assert state["done"] == 6
    assert state["error"] is None


def test_a_failure_keeps_the_stage_and_says_why():
    tracker = progress.Tracker(PID, progress.UPLOAD_STAGES)
    tracker.stage("notes")
    tracker.finish("Processing failed: boom")
    state = current()
    assert state["stage"] == "notes"
    assert state["error"] == "Processing failed: boom"
    assert state["finished"] is True


def test_without_an_id_nothing_is_recorded():
    progress.Tracker(None, progress.UPLOAD_STAGES).stage("notes")
    progress.Tracker("not-hex", progress.UPLOAD_STAGES).stage("notes")
    assert progress._entries == {}


def test_finished_entries_expire():
    clock = Clock()
    progress.Tracker(PID, progress.UPLOAD_STAGES, clock=clock).finish()
    clock.now += progress.KEEP_SECONDS + 1
    assert progress.snapshot(PID, clock=clock) is None


def test_the_route(client):
    assert client.get(f"/progress/{PID}").status_code == 404
    assert client.get("/progress/../etc").status_code == 404
    body = {"file": (io.BytesIO(wav_bytes()), "song.wav")}
    client.post(
        "/upload",
        data=body,
        content_type="multipart/form-data",
        headers={"X-Progress-Id": PID, "Accept": "application/json"},
    )
    state = client.get(f"/progress/{PID}").get_json()
    assert state["finished"] is True
    assert state["done"] == state["total"] == len(progress.UPLOAD_STAGES)


def test_progress_is_visible_while_analysing(client, monkeypatch):
    seen = {}

    def slow_analyse(audio_path, folder, report):
        report("rhythm")
        seen["mid"] = progress.snapshot(PID)
        analysis.write_tracks(folder, {"kick": [(0.5, "kick")]})
        return {"bpm": 120.0, "confidence": 0.5, "key": "A minor"}

    monkeypatch.setattr(analysis, "analyse", slow_analyse)
    body = {"file": (io.BytesIO(wav_bytes()), "song.wav")}
    client.post(
        "/upload", data=body, content_type="multipart/form-data", headers={"X-Progress-Id": PID}
    )
    assert seen["mid"]["stage"] == "rhythm"
    assert seen["mid"]["finished"] is False


def test_a_failed_analysis_reports_the_error(client, monkeypatch):
    def broken(audio_path, folder, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(analysis, "analyse", broken)
    body = {"file": (io.BytesIO(wav_bytes()), "song.wav")}
    client.post(
        "/upload", data=body, content_type="multipart/form-data", headers={"X-Progress-Id": PID}
    )
    assert client.get(f"/progress/{PID}").get_json()["error"] == "Processing failed: boom"
