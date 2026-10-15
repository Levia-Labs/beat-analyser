import io
import zipfile

from conftest import wav_bytes


def upload(client, name="song.wav", data=None, **headers):
    body = {"file": (io.BytesIO(data or wav_bytes()), name)}
    return client.post("/upload", data=body, content_type="multipart/form-data", headers=headers)


def test_index_has_csp_and_no_inline_script(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert b"<script>" not in response.data


def test_upload_returns_zip_and_caches(client, upload_root):
    response = upload(client)
    assert response.status_code == 200
    assert "song_tracks.zip" in response.headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(response.data)) as z:
        assert set(z.namelist()) == {"song/audio.wav", "song/meta.json", "song/kick.csv"}

    (upload_root / "song" / "kick.csv").unlink()
    assert upload(client).status_code == 200  # served from the cached ZIP
    assert not (upload_root / "song" / "kick.csv").exists()


def test_upload_traversal_stays_inside_root(client, upload_root):
    assert upload(client, name="../../evil.wav").status_code == 200
    assert (upload_root / "evil").is_dir()
    assert not (upload_root.parent / "evil").exists()


def test_upload_rejects_bad_input(client):
    assert upload(client, name="notes.txt").status_code == 400
    assert upload(client, name="日本.wav").status_code == 400
    assert client.post("/upload", data={}).status_code == 400


def test_upload_too_large(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    try:
        response = upload(client, data=b"0" * 4096)
    finally:
        client.application.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
    assert response.status_code == 413
    assert "error" in response.get_json()


def test_cross_origin_post_refused(client):
    assert upload(client, Origin="https://evil.example").status_code == 403
    assert upload(client, **{"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/repair", headers={"Origin": "https://evil.example"}).status_code == 403
    assert upload(client, Origin="http://localhost").status_code == 200


def test_folders_and_download(client):
    upload(client)
    listing = client.get("/folders").get_json()
    assert "song" in listing
    assert client.get("/download/song").status_code == 200
    assert client.get("/download/missing").status_code == 404
    assert client.get("/download/..").status_code == 404


def test_download_yt_rejects_foreign_url(client):
    response = client.post("/download_yt", json={"url": "https://evil.example/?v=dQw4w9WgXcQ"})
    assert response.status_code == 400
    assert client.post("/download_yt", json={}).status_code == 400
    assert client.post("/download_yt", data="nope").status_code == 400


def test_repair_rebuilds(client, upload_root):
    upload(client)
    (upload_root / "song" / "kick.csv").unlink()
    result = client.post("/repair").get_json()["repaired_projects"]
    assert result == [{"folder": "song", "zip": "song_tracks.zip"}]
    assert (upload_root / "song" / "kick.csv").exists()
