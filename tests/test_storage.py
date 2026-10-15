import zipfile

import pytest

import storage


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("song.mp3", "song"),
        ("My Song.wav", "My_Song"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.wav", "evil"),
        ("track.v2.flac", "track.v2"),
        ("café.mp3", "cafe"),
    ],
)
def test_project_name_is_sanitised(filename, expected):
    assert storage.project_name(filename) == expected


@pytest.mark.parametrize("filename", ["", "..", "../", ".wav", "日本.mp3"])
def test_project_name_rejects_unusable(filename):
    with pytest.raises(ValueError):
        storage.project_name(filename)


@pytest.mark.parametrize("name", ["..", ".hidden", "a/b", "a\\b", "", "x" * 101])
def test_invalid_names(name):
    assert not storage.is_valid_name(name)


def test_package_and_list(tmp_path):
    folder = storage.project_dir(tmp_path, "song")
    folder.mkdir()
    (folder / "kick.csv").write_text("0.5000,kick\n")
    storage.save_meta(folder, "song")

    archive = storage.package(tmp_path, "song")

    assert archive == tmp_path / "song_tracks.zip"
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["song/kick.csv", "song/meta.json"]
    assert not list(tmp_path.glob("*.part"))

    listing = storage.list_projects(tmp_path)
    assert listing["song"]["meta.json"]["content"] == {"name": "song", "key": "C", "bpm": 120}
    assert listing["song"]["kick.csv"]["size"] > 0


def test_clear_tracks_keeps_audio_and_meta(tmp_path):
    for name in ("audio.wav", "meta.json", "kick.csv", "bass.csv"):
        (tmp_path / name).write_text("x")
    storage.clear_tracks(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.wav", "meta.json"]
