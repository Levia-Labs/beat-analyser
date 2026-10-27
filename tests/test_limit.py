"""At most MAX_ANALYSES analyses at once; a cached result does not count."""

import threading
import zipfile

import pytest
from test_app import upload

import app as app_module
import pipeline
import storage
from conftest import wav_bytes

MESSAGE = "Too many analyses running; try again shortly"


@pytest.fixture
def full(monkeypatch):
    """Both slots held, as by two running analyses."""
    slots = app_module.ANALYSES
    held = 0
    while slots.acquire(blocking=False):
        held += 1
    yield
    for _ in range(held):
        slots.release()


def test_a_third_upload_is_429_and_writes_nothing(client, upload_root, full):
    response = upload(client)
    assert response.status_code == 429
    assert response.get_json() == {"error": MESSAGE}
    assert list(upload_root.iterdir()) == []


def test_a_third_upload_over_a_project_leaves_it_alone(client, upload_root):
    upload(client)
    before = sorted(p.name for p in (upload_root / "song").iterdir())
    slots = app_module.ANALYSES
    held = 0
    while slots.acquire(blocking=False):
        held += 1
    try:
        assert upload(client, data=wav_bytes(0.2)).status_code == 429
    finally:
        for _ in range(held):
            slots.release()
    assert sorted(p.name for p in (upload_root / "song").iterdir()) == before
    assert (upload_root / "song" / "audio.wav").read_bytes() == wav_bytes()


def test_a_repeat_at_the_cap_is_refused_because_it_must_place_and_hash(client, upload_root):
    # ADR 0014: the new audio is placed and hashed before the cache can be served.
    assert upload(client).status_code == 200
    slots = app_module.ANALYSES
    held = 0
    while slots.acquire(blocking=False):
        held += 1
    try:
        assert upload(client).status_code == 429
    finally:
        for _ in range(held):
            slots.release()


def test_a_full_cap_refuses_before_audio_is_placed(client, upload_root):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    placed = []
    with pytest.raises(pipeline.TooBusy):
        pipeline.build(upload_root, "song", lambda path: placed.append(path), slots=slots)
    assert placed == []


def test_a_cached_repeat_below_the_cap_releases_its_slot(client, upload_root):
    slots = threading.BoundedSemaphore(1)

    def place(path):
        path.write_bytes(wav_bytes())

    first = pipeline.build(upload_root, "song", place, slots=slots)
    again = pipeline.build(upload_root, "song", place, slots=slots)
    assert again == first
    assert slots.acquire(blocking=False)
    slots.release()


def test_a_stale_staged_file_is_removed_and_never_packaged(client, upload_root):
    folder = upload_root / "song"
    folder.mkdir(parents=True)
    (folder / storage.STAGED_NAME).write_bytes(b"left over from a crash")

    def place(path):
        path.write_bytes(wav_bytes())

    archive = pipeline.build(upload_root, "song", place)
    with zipfile.ZipFile(archive) as z:
        assert not any(n.endswith(storage.STAGED_NAME) for n in z.namelist())
    assert not (folder / storage.STAGED_NAME).exists()


def test_the_slot_is_released_after_a_failure(client, upload_root, monkeypatch):
    import analysis

    def boom(audio_path, folder, progress=None):
        raise RuntimeError("x")

    monkeypatch.setattr(analysis, "analyse", boom)
    for _ in range(4):
        assert upload(client).status_code == 500
