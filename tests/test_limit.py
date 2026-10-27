"""At most MAX_ANALYSES analyses at once; a cached result does not count."""

import pytest
from test_app import upload

import app as app_module
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


def test_a_cached_project_is_served_at_the_cap(client, upload_root):
    assert upload(client).status_code == 200
    slots = app_module.ANALYSES
    held = 0
    while slots.acquire(blocking=False):
        held += 1
    try:
        assert upload(client).status_code == 200
    finally:
        for _ in range(held):
            slots.release()


def test_the_slot_is_released_after_a_failure(client, upload_root, monkeypatch):
    import analysis

    def boom(audio_path, folder, progress=None):
        raise RuntimeError("x")

    monkeypatch.setattr(analysis, "analyse", boom)
    for _ in range(4):
        assert upload(client).status_code == 500
