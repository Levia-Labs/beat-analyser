from pathlib import Path
from typing import ClassVar

import pytest

import youtube

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"http://m.youtube.com/watch?list=x&v={VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"  https://www.youtube.com/watch?v={VID}  ",
    ],
)
def test_video_id_accepts(url):
    assert youtube.video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        f"https://evil.example/?v={VID}",
        f"https://evil.example/?https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com.evil.example/watch?v={VID}",
        f"ftp://www.youtube.com/watch?v={VID}",
        f"file:///etc/passwd?v={VID}",
        "https://www.youtube.com/watch?v=short",
        f"https://www.youtube.com/watch?v={VID}x",
        "https://www.youtube.com/watch?v=../../../etc",
    ],
)
def test_video_id_rejects(url):
    with pytest.raises(ValueError):
        youtube.video_id(url)


class _FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: download=True feeds the progress hooks."""

    statuses: ClassVar[list[dict]] = []

    def __init__(self, params):
        self.hooks = params["progress_hooks"]
        self.folder = Path(params["outtmpl"]).parent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        if download:
            for status in self.statuses:
                for hook in self.hooks:
                    hook(status)
        return {"duration": 10}

    def prepare_filename(self, info):
        path = self.folder / "source.webm"
        path.write_bytes(b"x")
        return str(path)


def _download(monkeypatch, tmp_path, statuses, on_progress):
    import yt_dlp

    _FakeYoutubeDL.statuses = statuses
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    return youtube.download_audio(VID, tmp_path, 600, 10**6, on_progress)


def test_download_progress_fractions(monkeypatch, tmp_path):
    seen: list[float] = []
    statuses = [
        {"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100},
        {"status": "downloading", "downloaded_bytes": 50, "total_bytes_estimate": 100},
        {"status": "downloading", "downloaded_bytes": 60},
        {"status": "finished", "downloaded_bytes": 100, "total_bytes": 100},
    ]
    _download(monkeypatch, tmp_path, statuses, seen.append)
    assert seen == [0.25, 0.5]


def test_download_without_progress_callback(monkeypatch, tmp_path):
    statuses = [{"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2}]
    assert _download(monkeypatch, tmp_path, statuses, None).name == "source.webm"


def test_canonical_url():
    assert youtube.canonical_url(VID) == f"https://www.youtube.com/watch?v={VID}"


@pytest.mark.parametrize(
    ("title", "name"),
    [
        (
            "Rick Astley - Never Gonna Give You Up (Official Video)",
            "Rick_Astley_-_Never_Gonna_Give_You_Up_Official_Video",
        ),
        ("Beyoncé \N{EN DASH} Halo", "Beyonce_Halo"),
        ("../../etc/passwd", "etc_passwd"),
        ("--- intro", "intro"),
        ("x" * 150, "x" * 100),
        # Nothing usable left: the project falls back to the video ID.
        ("日本語のタイトル", VID),
        ("", VID),
        (None, VID),
    ],
)
def test_project_name_from_title(title, name):
    assert youtube.project_name(title, VID) == name


def test_project_name_of_an_id_with_a_leading_dash():
    assert youtube.project_name(None, "-abcdefghij") == "yt-abcdefghij"


class _FakeInfoDL:
    """Stands in for yt_dlp.YoutubeDL when only the metadata is fetched."""

    info: ClassVar[dict | Exception] = {}
    urls: ClassVar[list[str]] = []

    def __init__(self, params):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        assert not download
        self.urls.append(url)
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


@pytest.mark.parametrize(
    ("info", "title"),
    [
        ({"title": "Never Gonna Give You Up"}, "Never Gonna Give You Up"),
        ({}, None),
        ("error", None),
    ],
)
def test_title(monkeypatch, info, title):
    import yt_dlp
    from yt_dlp.utils import DownloadError

    _FakeInfoDL.info = DownloadError("private video") if info == "error" else info
    _FakeInfoDL.urls = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeInfoDL)
    assert youtube.title(VID) == title
    assert _FakeInfoDL.urls == [youtube.canonical_url(VID)]
