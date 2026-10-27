"""Deleting and renaming projects from the SPA."""

import io
import json
import shutil
import zipfile

import app as app_module
import pipeline
import storage
from conftest import wav_bytes

JSON = {"Accept": "application/json"}


def make(client, name="song"):
    body = {"file": (io.BytesIO(wav_bytes()), f"{name}.wav")}
    response = client.post("/upload", data=body, content_type="multipart/form-data", headers=JSON)
    assert response.status_code == 200


def test_delete_removes_folder_and_zip(client, upload_root):
    make(client)
    response = client.delete("/projects/song")
    assert response.status_code == 204
    assert not (upload_root / "song").exists()
    assert not storage.zip_path(upload_root, "song").exists()


def test_delete_unknown_or_invalid_is_404(client):
    assert client.delete("/projects/nope").status_code == 404
    assert client.delete("/projects/..%2Fetc").status_code == 404


def test_delete_while_analysing_is_409(client, upload_root):
    make(client)
    with pipeline._lock("song"):
        assert client.delete("/projects/song").status_code == 409
    assert (upload_root / "song").is_dir()


def test_cross_origin_delete_is_refused(client, upload_root):
    make(client)
    response = client.delete("/projects/song", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert (upload_root / "song").is_dir()


def test_rename_moves_meta_and_zip(client, upload_root):
    make(client)
    response = client.post("/projects/song/rename", json={"to": "tune"})
    assert response.get_json() == {"name": "tune"}
    assert not (upload_root / "song").exists()
    assert not storage.zip_path(upload_root, "song").exists()
    meta = json.loads((upload_root / "tune" / storage.META_NAME).read_text(encoding="utf-8"))
    assert meta["name"] == "tune"
    names = zipfile.ZipFile(storage.zip_path(upload_root, "tune")).namelist()
    assert names and all(name.startswith("tune/") for name in names)


def test_rename_refuses_bad_names_and_collisions(client):
    make(client)
    make(client, "other")
    assert client.post("/projects/song/rename", json={"to": "../x"}).status_code == 400
    assert client.post("/projects/song/rename", json={}).status_code == 400
    assert client.post("/projects/song/rename", json={"to": "other"}).status_code == 409
    assert client.post("/projects/nope/rename", json={"to": "x"}).status_code == 404


def test_rename_to_the_same_name_is_a_no_op(client):
    make(client)
    assert client.post("/projects/song/rename", json={"to": "song"}).get_json() == {"name": "song"}


def test_rename_while_analysing_is_409(client, upload_root):
    make(client)
    with pipeline._lock("song"):
        assert client.post("/projects/song/rename", json={"to": "tune"}).status_code == 409
    assert (upload_root / "song").is_dir()


def test_rename_survives_corrupt_meta(client, upload_root):
    make(client)
    (upload_root / "song" / storage.META_NAME).write_text("{not json", encoding="utf-8")
    response = client.post("/projects/song/rename", json={"to": "tune"})
    assert response.get_json() == {"name": "tune"}
    assert (upload_root / "tune").is_dir()
    assert not (upload_root / "song").exists()
    assert storage.zip_path(upload_root, "tune").is_file()
    assert not storage.zip_path(upload_root, "song").exists()


def test_rename_packaging_failure_keeps_the_old_project(client, upload_root, monkeypatch):
    make(client)

    def boom(root, name):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "package", boom)
    monkeypatch.setitem(app_module.app.config, "PROPAGATE_EXCEPTIONS", False)
    response = client.post("/projects/song/rename", json={"to": "tune"})
    assert response.status_code == 500
    assert (upload_root / "song").is_dir()
    assert not (upload_root / "tune").exists()
    assert storage.zip_path(upload_root, "song").is_file()
    meta = json.loads((upload_root / "song" / storage.META_NAME).read_text(encoding="utf-8"))
    assert meta["name"] == "song"


def test_rename_zip_only_project_is_404(client, upload_root):
    make(client)
    shutil.rmtree(upload_root / "song")
    assert client.post("/projects/song/rename", json={"to": "tune"}).status_code == 404
    assert storage.zip_path(upload_root, "song").is_file()


def test_rename_refuses_malformed_bodies(client):
    make(client)
    assert client.post("/projects/song/rename", json=[]).status_code == 400
    assert client.post("/projects/song/rename", json="tune").status_code == 400
    assert client.post("/projects/song/rename", json={"to": None}).status_code == 400
    assert client.post("/projects/song/rename", json={"to": 5}).status_code == 400


def test_try_lock_releases_after_busy():
    with pipeline._lock("b"):
        try:
            with pipeline.try_lock("a", "b"):
                raise AssertionError("should be busy")
        except pipeline.Busy:
            pass
    assert pipeline._lock("a").acquire(blocking=False)
    pipeline._lock("a").release()
