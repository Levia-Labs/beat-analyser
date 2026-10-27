"""The MCP service's client for the web app's analysis endpoints. urlopen is replaced, so
no request leaves the process."""

import http.client
import io
import json
import urllib.error
from email.message import Message

import pytest

import backend
import youtube


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    requests = []

    def urlopen(req, timeout):
        requests.append((req, timeout))
        return Response()

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)
    return requests


def refuse_with(monkeypatch, exc):
    def urlopen(req, timeout):
        raise exc

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)


def http_error(status, body):
    return urllib.error.HTTPError(
        "http://backend:8000/upload", status, "error", Message(), io.BytesIO(body)
    )


def test_upload_posts_multipart_with_the_filename(sent):
    backend.upload("http://backend:8000", "song.mp3", b"ID3data")
    req, timeout = sent[0]
    assert req.full_url == "http://backend:8000/upload"
    assert req.get_method() == "POST"
    assert timeout == backend.TIMEOUT
    content_type = req.get_header("Content-type")
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("=", 1)[1]
    assert req.data.startswith(f"--{boundary}\r\n".encode())
    assert b'name="file"; filename="song.mp3"' in req.data
    assert b"\r\n\r\nID3data\r\n" in req.data
    assert req.data.endswith(f"--{boundary}--\r\n".encode())


def test_download_yt_posts_the_canonical_url(sent):
    backend.download_yt("http://backend:8000", "dQw4w9WgXcQ")
    req, _ = sent[0]
    assert req.full_url == "http://backend:8000/download_yt"
    assert req.get_header("Content-type") == "application/json"
    assert json.loads(req.data) == {"url": youtube.canonical_url("dQw4w9WgXcQ")}


def test_a_refusal_carries_the_web_app_message(monkeypatch):
    refuse_with(monkeypatch, http_error(400, b'{"error": "Unsupported file type"}'))
    with pytest.raises(backend.BackendError, match=r"^400: Unsupported file type$"):
        backend.upload("http://backend:8000", "song.mp3", b"x")


def test_a_non_json_error_page_is_quoted(monkeypatch):
    refuse_with(monkeypatch, http_error(502, b"<html>Bad Gateway</html>"))
    with pytest.raises(backend.BackendError, match=r"^502: <html>Bad Gateway</html>$"):
        backend.upload("http://backend:8000", "song.mp3", b"x")


def test_an_unreachable_backend_is_a_backend_error(monkeypatch):
    refuse_with(monkeypatch, urllib.error.URLError("Connection refused"))
    with pytest.raises(backend.BackendError, match="unreachable"):
        backend.download_yt("http://backend:8000", "dQw4w9WgXcQ")


def test_a_dropped_connection_is_a_backend_error(monkeypatch):
    refuse_with(
        monkeypatch,
        http.client.RemoteDisconnected("Remote end closed connection without response"),
    )
    with pytest.raises(backend.BackendError, match="unreachable"):
        backend.upload("http://backend:8000", "song.mp3", b"x")


def test_a_timeout_is_a_backend_error(monkeypatch):
    refuse_with(monkeypatch, TimeoutError("timed out"))
    with pytest.raises(backend.BackendError, match="unreachable"):
        backend.upload("http://backend:8000", "song.mp3", b"x")
