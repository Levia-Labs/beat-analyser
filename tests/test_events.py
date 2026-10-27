"""Reading what an analysis wrote, shared by the web app and the MCP tools."""

import json

import events
import storage


def make_project(root, name="song", built=True):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "beats.csv").write_text("0.5,131\n1,131\nbad,row\n2,131\n", encoding="utf-8")
    (folder / "kick.csv").write_text("0.25,128\n", encoding="utf-8")
    (folder / "melody.csv").write_text("1.5,69\n", encoding="utf-8")
    (folder / "extra.csv").write_text("", encoding="utf-8")
    (folder / "labels.json").write_text(
        json.dumps({"128": "kick", "131": "beat"}), encoding="utf-8"
    )
    (folder / "meta.json").write_text(
        json.dumps({"name": name, "key": "A minor", "bpm": 120.0}), encoding="utf-8"
    )
    if built:
        storage.zip_path(root, name).write_bytes(b"PK")
    return folder


def test_tracks_follow_the_lane_order_then_the_rest(tmp_path):
    folder = make_project(tmp_path)
    assert list(events.tracks(folder)) == ["kick", "beats", "melody", "extra"]


def test_tracks_of_a_missing_folder_is_empty(tmp_path):
    assert events.tracks(tmp_path / "nope") == {}


def test_rows_skip_malformed_lines(tmp_path):
    folder = make_project(tmp_path)
    assert list(events.rows(folder / "beats.csv")) == [(0.5, 131), (1.0, 131), (2.0, 131)]


def test_labels_and_meta(tmp_path):
    folder = make_project(tmp_path)
    assert events.labels(folder) == {128: "kick", 131: "beat"}
    assert events.meta(folder) == {"name": "song", "key": "A minor", "bpm": 120.0}
    (folder / "labels.json").unlink()
    (folder / "meta.json").write_text("{not json", encoding="utf-8")
    assert events.labels(folder) == {}
    assert events.meta(folder) is None


def test_summary(tmp_path):
    folder = make_project(tmp_path)
    assert events.summary(folder)[1] == {"track": "beats", "events": 3, "first": 0.5, "last": 2.0}
    assert events.summary(folder)[3] == {"track": "extra", "events": 0, "first": None, "last": None}


def test_project_list(tmp_path):
    make_project(tmp_path)
    (tmp_path / "half").mkdir()
    assert events.project_list(tmp_path) == [
        {"name": "half", "done": False, "meta": None},
        {"name": "song", "done": True, "meta": {"name": "song", "key": "A minor", "bpm": 120.0}},
    ]
    assert events.project_list(tmp_path / "missing") == []


def test_project_events(tmp_path):
    make_project(tmp_path)
    result = events.project_events(tmp_path, "song")
    assert result["name"] == "song"
    assert result["meta"]["bpm"] == 120.0
    assert result["labels"] == {"128": "kick", "131": "beat"}
    assert list(result["tracks"]) == ["kick", "beats", "melody", "extra"]
    assert result["tracks"]["beats"] == [[0.5, 131], [1.0, 131], [2.0, 131]]
