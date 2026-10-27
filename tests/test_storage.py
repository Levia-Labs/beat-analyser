import os
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
    (folder / "kick.csv").write_text("0.5000,kick\n", encoding="utf-8")
    storage.save_meta(folder, "song", key="A minor", bpm=128.0)

    archive = storage.package(tmp_path, "song")

    assert archive == tmp_path / "song_tracks.zip"
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["song/kick.csv", "song/meta.json"]
    assert not list(tmp_path.glob("*.part"))

    listing = storage.list_projects(tmp_path)
    meta = listing["song"]["meta.json"]["content"]
    assert meta == {"name": "song", "key": "A minor", "bpm": 128.0}
    assert listing["song"]["kick.csv"]["size"] > 0


def test_clear_tracks_keeps_audio_and_meta(tmp_path):
    for name in ("audio.wav", "meta.json", "kick.csv", "bass.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    storage.clear_tracks(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.wav", "meta.json"]


def _project(root, name, built=True, meta=True):
    folder = storage.project_dir(root, name)
    folder.mkdir()
    (folder / "kick.csv").write_text("0.5,128\n", encoding="utf-8")
    if meta:
        storage.save_meta(folder, name, key="A minor", bpm=128.0)
    if built:
        storage.package(root, name)
    return folder


def test_rename_built_project_moves_folder_and_zip(tmp_path):
    _project(tmp_path, "old")
    storage.rename_project(tmp_path, "old", "new")
    new_meta = (tmp_path / "new" / "meta.json").read_text(encoding="utf-8")
    assert '"name": "new"' in new_meta
    assert storage.zip_path(tmp_path, "new").is_file()
    with zipfile.ZipFile(storage.zip_path(tmp_path, "new")) as z:
        assert "new/kick.csv" in z.namelist()


def test_rename_keeps_meta_mtime_so_created_does_not_jump(tmp_path):
    folder = _project(tmp_path, "old")
    os.utime(folder / "meta.json", ns=(1_600_000_000 * 10**9, 1_600_000_000 * 10**9))
    storage.rename_project(tmp_path, "old", "new")
    assert (tmp_path / "new" / "meta.json").stat().st_mtime_ns == 1_600_000_000 * 10**9


def test_package_skips_the_staged_audio(tmp_path):
    folder = _project(tmp_path, "song", built=False)
    (folder / storage.STAGED_NAME).write_bytes(b"x")
    with zipfile.ZipFile(storage.package(tmp_path, "song")) as z:
        assert sorted(z.namelist()) == ["song/kick.csv", "song/meta.json"]


def test_rename_keeps_corrupt_meta_as_is(tmp_path):
    folder = _project(tmp_path, "old", built=False)
    (folder / "meta.json").write_text("{not json", encoding="utf-8")
    storage.rename_project(tmp_path, "old", "new")
    assert (tmp_path / "new" / "meta.json").read_text(encoding="utf-8") == "{not json"
    assert not storage.zip_path(tmp_path, "new").exists()


def test_rename_rolls_back_when_package_fails(tmp_path, monkeypatch):
    _project(tmp_path, "old")
    original = (tmp_path / "old" / "meta.json").read_text(encoding="utf-8")

    def boom(root, name):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "package", boom)
    with pytest.raises(OSError):
        storage.rename_project(tmp_path, "old", "new")
    assert not (tmp_path / "new").exists()
    assert (tmp_path / "old" / "meta.json").read_text(encoding="utf-8") == original


def test_delete_folder_only_and_zip_only(tmp_path):
    _project(tmp_path, "a", built=False)
    storage.delete_project(tmp_path, "a")
    assert not (tmp_path / "a").exists()

    _project(tmp_path, "b")
    import shutil

    shutil.rmtree(tmp_path / "b")
    storage.delete_project(tmp_path, "b")
    assert not storage.zip_path(tmp_path, "b").exists()
