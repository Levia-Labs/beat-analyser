"""The MCP tools, called as plain functions: jobs run inline, the web app is a stub, and
projects live in a temporary root."""

import base64
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import backend
import mcp_tools
import storage
from jobs import Jobs


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(mcp_tools, "jobs", Jobs(submit=lambda work: work()))
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(backend, "upload", lambda *args: seen.append(("upload", *args)))
    monkeypatch.setattr(backend, "download_yt", lambda *args: seen.append(("yt", *args)))
    return seen


def make_project(root, name="song", built=True):
    folder = root / name
    folder.mkdir()
    (folder / "beats.csv").write_text("0.5,131\n1,131\n1.5,131\nbad,row\n2,131\n", encoding="utf-8")
    (folder / "kick.csv").write_text("", encoding="utf-8")
    (folder / "labels.json").write_text(json.dumps({"131": "beat"}), encoding="utf-8")
    meta = {"name": name, "key": "A minor", "bpm": 120.0}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if built:
        storage.zip_path(root, name).write_bytes(b"PK")
    return folder


def test_analyse_audio_uploads_under_the_sanitised_name(root, calls):
    result = mcp_tools.analyse_audio("C:\\music\\My Song.MP3", b64(b"ID3data"))
    assert result == {"name": "My_Song", "status": "done"}
    assert calls == [("upload", mcp_tools.BACKEND_URL, "My_Song.mp3", b"ID3data")]


def test_wrapped_base64_is_accepted(root, calls):
    data = bytes(range(256)) * 2
    mcp_tools.analyse_audio("song.wav", base64.encodebytes(data).decode())
    assert calls[0][3] == data


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("notes.txt", b64(b"x"), "Unsupported file type"),
        (".wav", b64(b"x"), "Unsupported file type"),
        ("song.wav", "@@@@", "not valid base64"),
        ("song.wav", "", "Empty audio"),
    ],
)
def test_analyse_audio_refusals(root, calls, filename, payload, message):
    with pytest.raises(ToolError, match=message):
        mcp_tools.analyse_audio(filename, payload)
    assert calls == []


def test_analyse_audio_refuses_more_than_the_limit(root, calls, monkeypatch):
    monkeypatch.setattr(mcp_tools, "MAX_BYTES", 4)
    with pytest.raises(ToolError, match="larger than"):
        mcp_tools.analyse_audio("song.wav", b64(b"12345"))
    assert calls == []


def test_analyse_youtube_reduces_the_url_to_an_id(root, calls):
    result = mcp_tools.analyse_youtube("https://youtu.be/dQw4w9WgXcQ")
    assert result == {"name": "dQw4w9WgXcQ", "status": "done"}
    assert calls == [("yt", mcp_tools.BACKEND_URL, "dQw4w9WgXcQ")]


def test_analyse_youtube_refuses_a_foreign_host(root, calls):
    with pytest.raises(ToolError):
        mcp_tools.analyse_youtube("https://example.com/watch?v=dQw4w9WgXcQ")
    assert calls == []


def test_a_full_registry_is_a_tool_error(root, monkeypatch, calls):
    monkeypatch.setattr(mcp_tools, "jobs", Jobs(submit=lambda work: None, max_pending=1))
    mcp_tools.analyse_audio("a.wav", b64(b"x"))
    with pytest.raises(ToolError, match="pending"):
        mcp_tools.analyse_audio("b.wav", b64(b"x"))


def test_a_full_registry_refuses_before_decoding(root, monkeypatch, calls):
    monkeypatch.setattr(mcp_tools, "jobs", Jobs(submit=lambda work: None, max_pending=1))
    mcp_tools.analyse_audio("a.wav", b64(b"x"))
    with pytest.raises(ToolError, match="pending"):
        mcp_tools.analyse_audio("b.wav", "!!not base64!!")


def test_a_refused_upload_is_a_failed_job(root, monkeypatch):
    def refuse(*args):
        raise backend.BackendError("400: Unsupported file type")

    monkeypatch.setattr(backend, "upload", refuse)
    mcp_tools.analyse_audio("song.wav", b64(b"x"))
    assert mcp_tools.project_status("song") == {
        "name": "song",
        "status": "failed",
        "error": "400: Unsupported file type",
    }


def test_a_project_built_by_the_web_ui_is_done(root):
    make_project(root)
    status = mcp_tools.project_status("song")
    assert status["status"] == "done"
    assert status["meta"]["key"] == "A minor"
    assert status["tracks"] == [
        {"track": "kick", "events": 0, "first": None, "last": None},
        {"track": "beats", "events": 4, "first": 0.5, "last": 2.0},
    ]


def test_a_folder_without_its_zip_is_unknown(root):
    make_project(root, built=False)
    assert mcp_tools.project_status("song") == {"name": "song", "status": "unknown"}


@pytest.mark.parametrize("tool", ["project_status", "get_events"])
def test_a_traversal_name_is_refused(root, tool):
    args = ("../etc",) if tool == "project_status" else ("../etc", "kick")
    with pytest.raises(ToolError, match="Invalid project name"):
        getattr(mcp_tools, tool)(*args)


def test_list_projects(root):
    make_project(root)
    (root / "half").mkdir()
    assert mcp_tools.list_projects() == {
        "projects": [
            {"name": "half", "done": False, "meta": None},
            {
                "name": "song",
                "done": True,
                "meta": {"name": "song", "key": "A minor", "bpm": 120.0},
            },
        ]
    }


def test_list_projects_without_a_root(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "UPLOAD_ROOT", tmp_path / "missing")
    assert mcp_tools.list_projects() == {"projects": []}


def test_get_events_window(root):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", start=1, end=2)
    assert result == {
        "name": "song",
        "track": "beats",
        "rows": [[1.0, "beat"], [1.5, "beat"]],
        "truncated": False,
    }


def test_get_events_truncates_at_the_limit(root):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", limit=2)
    assert result["rows"] == [[0.5, "beat"], [1.0, "beat"]]
    assert result["truncated"] is True


@pytest.mark.parametrize("limit", [0, -5])
def test_get_events_clamps_the_limit(root, limit):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", limit=limit)
    assert result["rows"] == [[0.5, "beat"]]
    assert result["truncated"] is True


def test_get_events_empty_window(root):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", start=2, end=1)
    assert result["rows"] == []
    assert result["truncated"] is False


def test_get_events_without_labels_returns_the_id(root):
    folder = make_project(root)
    (folder / "labels.json").unlink()
    assert mcp_tools.get_events("song", "beats", limit=1)["rows"] == [[0.5, "131"]]


def test_get_events_refuses_an_unknown_track(root):
    make_project(root)
    with pytest.raises(ToolError, match="tracks: kick, beats"):
        mcp_tools.get_events("song", "../meta")
