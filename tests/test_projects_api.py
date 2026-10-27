"""GET /projects: filter, sort and page, and the duration pipeline.build stores."""

import os
import wave
from pathlib import Path

import pytest

import catalog
import pipeline
import storage
from conftest import wav_bytes


def make(root, name, key="A minor", bpm=120.0, duration=None, built=True, created=1_700_000_000):
    folder = root / name
    folder.mkdir(parents=True)
    storage.save_meta(folder, name, key=key, bpm=bpm, duration=duration)
    os.utime(folder / storage.META_NAME, (created, created))
    (folder / "kick.csv").write_text("0.5,128\n", encoding="utf-8")
    if built:
        storage.package(root, name)


@pytest.fixture
def seeded(upload_root):
    upload_root.mkdir(parents=True)
    make(upload_root, "alpha", "A minor", 100.0, 10.5, created=1_700_000_100)
    make(upload_root, "Beta", "C major", 140.0, 30.0, created=1_700_000_300)
    make(upload_root, "gamma", "A minor", 120.0, None, built=False, created=1_700_000_200)
    (upload_root / "legacy").mkdir()
    (upload_root / "legacy" / "kick.csv").write_text("0.5,128\n", encoding="utf-8")


def names(client, query=""):
    body = client.get(f"/projects?{query}").get_json()
    return [item["name"] for item in body["items"]]


def test_default_is_newest_first_with_shape(client, seeded):
    body = client.get("/projects").get_json()
    assert body["total"] == 4
    assert [item["name"] for item in body["items"]] == ["Beta", "gamma", "alpha", "legacy"]
    beta = body["items"][0]
    assert set(beta) == {"name", "done", "key", "bpm", "duration", "created", "size"}
    assert beta["done"] is True
    assert beta["created"] == "2023-11-14T22:18:20Z"
    assert beta["size"] > 0
    legacy = body["items"][3]
    assert (legacy["key"], legacy["bpm"], legacy["created"]) == (None, None, None)


def test_filters(client, seeded):
    assert names(client, "q=ET") == ["Beta"]
    assert names(client, "key=A minor") == ["gamma", "alpha"]
    assert names(client, "key=major") == ["Beta"]
    assert names(client, "key=minor") == ["gamma", "alpha"]
    assert names(client, "status=pending") == ["gamma", "legacy"]
    assert names(client, "status=built") == ["Beta", "alpha"]
    assert names(client, "bpm_min=110&bpm_max=130") == ["gamma"]
    assert names(client, "bpm_min=140") == ["Beta"]


def test_total_counts_the_filtered_set_not_the_page(client, seeded):
    body = client.get("/projects?key=minor&limit=1&offset=1").get_json()
    assert body["total"] == 2
    assert [item["name"] for item in body["items"]] == ["alpha"]


def test_sort_puts_unknown_last_in_both_orders(client, seeded):
    assert names(client, "sort=bpm&order=asc") == ["alpha", "gamma", "Beta", "legacy"]
    assert names(client, "sort=bpm&order=desc") == ["Beta", "gamma", "alpha", "legacy"]
    assert names(client, "sort=duration&order=asc")[-2:] == ["gamma", "legacy"]
    assert names(client, "sort=name&order=asc") == ["alpha", "Beta", "gamma", "legacy"]


@pytest.mark.parametrize(
    "query",
    [
        "sort=nope",
        "order=up",
        "status=maybe",
        "key=H flat",
        "bpm_min=abc",
        "bpm_max=nan",
        "offset=-1",
        "offset=x",
        "limit=0",
        "limit=201",
    ],
)
def test_invalid_values_are_400(client, seeded, query):
    response = client.get(f"/projects?{query}")
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_empty_root(client):
    assert client.get("/projects").get_json() == {"items": [], "total": 0}


def test_a_file_vanishing_during_stat_counts_as_zero_bytes(client, seeded, monkeypatch):
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.name == "kick.csv" and self.parent.name == "alpha":
            raise FileNotFoundError(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    body = client.get("/projects?sort=name&order=asc").get_json()
    assert [item["name"] for item in body["items"]] == ["alpha", "Beta", "gamma", "legacy"]
    alpha = body["items"][0]
    assert alpha["size"] > 0  # the ZIP and meta still count


def test_a_folder_vanishing_mid_listing_is_skipped_unless_a_zip_remains(
    client, seeded, monkeypatch
):
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self.name in {"alpha", "gamma"}:
            raise FileNotFoundError(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    # gamma has no ZIP and vanished: skipped. alpha keeps its ZIP: listed with the ZIP's size.
    assert names(client, "sort=name&order=asc") == ["alpha", "Beta", "legacy"]


def test_build_stores_the_duration(client, upload_root):
    def place(path):
        path.write_bytes(wav_bytes())

    pipeline.build(upload_root, "song", place)
    with wave.open(str(upload_root / "song" / storage.AUDIO_NAME)) as audio:
        expected = round(audio.getnframes() / audio.getframerate(), 2)
    (item,) = client.get("/projects").get_json()["items"]
    assert item["duration"] == expected


def test_parse_defaults():
    query = catalog.parse({})
    assert (query.sort, query.order, query.offset, query.limit) == ("created", "desc", 0, 50)
