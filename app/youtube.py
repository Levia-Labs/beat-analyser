"""YouTube URL parsing and audio download.

The URL a client sends is never handed to yt-dlp. It is reduced to an 11-character video ID
and rebuilt as a canonical youtube.com URL: an unanchored regex used to accept
`https://anything/?https://www.youtube.com/watch?v=...`, and yt-dlp would then fetch from
whichever host the string really named.
"""

import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


def video_id(url: str) -> str:
    """The video ID of a watch, shorts or youtu.be URL. Raises ValueError otherwise."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    candidate = ""
    if parts.scheme in ("http", "https"):
        if host == "youtu.be":
            candidate = parts.path.lstrip("/").split("/")[0]
        elif host in _HOSTS and parts.path == "/watch":
            candidate = parse_qs(parts.query).get("v", [""])[0]
        elif host in _HOSTS and parts.path.startswith("/shorts/"):
            candidate = parts.path.split("/")[2]
    if not _ID.fullmatch(candidate):
        raise ValueError("Invalid YouTube URL")
    return candidate


def canonical_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def download_audio(
    vid: str,
    folder: Path,
    max_seconds: int,
    max_bytes: int,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Download the best audio stream into folder. Returns the downloaded file.

    Raises ValueError for a video that is too long or too large.
    """
    from yt_dlp import YoutubeDL

    def hook(status: dict) -> None:
        # yt-dlp reports bytes; the total is sometimes only an estimate, sometimes absent.
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        if on_progress and status.get("status") == "downloading" and total:
            on_progress(status.get("downloaded_bytes", 0) / total)

    # A literal in the call, not a variable: yt-dlp types `params` as a TypedDict, and only a
    # literal is checked against it key by key.
    with YoutubeDL(
        {
            "format": "bestaudio/best",
            "outtmpl": str(folder / "source.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            # The container's root file system is read-only; ~/.cache is not writable.
            # yt-dlp documents False as "no cache"; typeshed's stub allows only str | None.
            "cachedir": False,  # pyright: ignore[reportArgumentType]
            "max_filesize": max_bytes,
            "progress_hooks": [hook],
        }
    ) as ydl:
        info = ydl.extract_info(canonical_url(vid), download=False)
        if (info.get("duration") or 0) > max_seconds:
            raise ValueError(f"Video longer than {max_seconds // 60} minutes")
        info = ydl.extract_info(canonical_url(vid), download=True)
        path = Path(ydl.prepare_filename(info))
    # yt-dlp skips an over-size download without raising.
    if not path.is_file():
        raise ValueError(f"Audio larger than {max_bytes // (1024 * 1024)} MB")
    return path
