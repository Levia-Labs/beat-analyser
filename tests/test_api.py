"""The JSON side of the web app, used by the SPA in web/."""

import io
import zipfile

from conftest import wav_bytes

JSON = {"Accept": "application/json"}


def upload(client, headers):
    body = {"file": (io.BytesIO(wav_bytes()), "song.wav")}
    return client.post("/upload", data=body, content_type="multipart/form-data", headers=headers)


def test_upload_answers_the_name_when_json_is_asked(client):
    response = upload(client, JSON)
    assert response.status_code == 200
    assert response.get_json() == {"name": "song"}


def test_upload_still_answers_the_zip_by_default(client):
    response = upload(client, {"Accept": "*/*"})
    assert response.status_code == 200
    # The body, not the mimetype: Windows registers application/x-zip-compressed.
    assert zipfile.is_zipfile(io.BytesIO(response.data))


def test_download_yt_answers_the_name_when_json_is_asked(client, monkeypatch):
    import youtube

    def fake_download(vid, folder, max_seconds, max_bytes, on_progress=None):
        source = folder / "source.wav"
        source.write_bytes(wav_bytes())
        return source

    monkeypatch.setattr(youtube, "download_audio", fake_download)
    response = client.post(
        "/download_yt", json={"url": "https://youtu.be/dQw4w9WgXcQ"}, headers=JSON
    )
    assert response.status_code == 200
    assert response.get_json() == {"name": "dQw4w9WgXcQ"}


def test_config(client):
    import app as app_module

    assert client.get("/config").get_json() == {
        "extensions": [".flac", ".m4a", ".mp3", ".ogg", ".wav"],
        "max_upload_mb": app_module.app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
    }


def test_projects_lists_done_and_meta(client):
    upload(client, JSON)
    body = client.get("/projects").get_json()
    (project,) = body["items"]
    assert body["total"] == 1
    assert project["name"] == "song"
    assert project["done"] is True
    assert project["key"] == "A minor"


def test_events_of_a_built_project(client):
    upload(client, JSON)
    body = client.get("/projects/song/events").get_json()
    assert body["name"] == "song"
    assert body["tracks"]["kick"] == [[0.5, 128]]


def test_events_of_a_project_without_zip_is_404(client, upload_root):
    (upload_root / "half").mkdir(parents=True)
    response = client.get("/projects/half/events")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Project not found"}


def test_events_refuse_a_traversal_name(client):
    assert client.get("/projects/..%2Fetc/events").status_code == 404
    assert client.get("/projects/.hidden/events").status_code == 404
