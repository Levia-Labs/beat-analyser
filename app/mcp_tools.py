"""The MCP tools: start an analysis, poll it, read what it produced (docs/mcp.md).

Plain functions, registered on the server in mcp_server.py, so tests call them directly.
Every argument is checked here before it reaches a job or the file system: names through
storage.is_valid_name, URLs through youtube.video_id, audio through check_audio. Nothing here
writes under UPLOAD_ROOT; the service mounts it read-only.
"""

import base64
import binascii
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

import backend
import events
import storage
import youtube
from jobs import Jobs, JobsFull

MAX_ROWS = 10000

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "uploads")).resolve()
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024

jobs = Jobs()


def check_audio(filename: str, data_base64: str, max_bytes: int) -> tuple[str, str, bytes]:
    """Project name, extension and bytes of an agent's upload. Raises ValueError.

    The length is checked before decoding, so an oversized payload is refused without a
    second copy in memory. Line breaks are allowed: the `base64` command wraps at 76.
    """
    extension = Path(filename.replace("\\", "/")).suffix.lower()
    if extension not in storage.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(storage.ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type; use {allowed}")
    name = storage.project_name(filename)
    limit = 4 * ((max_bytes + 2) // 3)
    # At most two line-break characters per 76 of data, plus a trailing one.
    if len(data_base64) > limit + limit // 38 + 2:
        raise ValueError(f"Audio larger than {max_bytes} bytes")
    compact = "".join(data_base64.split())
    if len(compact) > limit:
        raise ValueError(f"Audio larger than {max_bytes} bytes")
    try:
        data = base64.b64decode(compact, validate=True)
    except binascii.Error:
        raise ValueError("data_base64 is not valid base64") from None
    if not data:
        raise ValueError("Empty audio")
    if len(data) > max_bytes:
        raise ValueError(f"Audio larger than {max_bytes} bytes")
    return name, extension, data


def analyse_youtube(url: str) -> dict[str, Any]:
    """Start analysing a YouTube video. Returns at once with the project name (the video's
    title, or its ID when the title has nothing usable); poll project_status(name) until it
    is done or failed."""
    try:
        vid = youtube.video_id(url)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    name = youtube.name_for(vid)
    return _start(name, lambda: backend.download_yt(BACKEND_URL, vid, name))


def analyse_audio(filename: str, data_base64: str) -> dict[str, Any]:
    """Start analysing an audio file (wav, mp3, flac, ogg, m4a) sent as base64. Returns at
    once with the project name (the file's sanitised stem); poll project_status(name)."""
    try:
        # Before decoding: a refused call must not allocate up to MAX_BYTES, and every pending
        # job holds its bytes. A bad name is left for check_audio to report.
        try:
            room_name = storage.project_name(filename)
        except ValueError:
            room_name = None
        if room_name is not None:
            try:
                jobs.check_room(room_name)
            except JobsFull as exc:
                raise ToolError(str(exc)) from None
        name, extension, data = check_audio(filename, data_base64, MAX_BYTES)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    return _start(name, lambda: backend.upload(BACKEND_URL, f"{name}{extension}", data))


def project_status(name: str) -> dict[str, Any]:
    """State of a project: queued, running, failed (with error), done (with meta and the event
    count and time span of each track), or unknown."""
    _check_name(name)
    job = jobs.get(name)
    if job is not None and job.status != "done":
        result: dict[str, Any] = {"name": name, "status": job.status}
        if job.error:
            result["error"] = job.error
        return result
    # Only the ZIP means done: pipeline.build writes meta.json before it packages.
    if not storage.zip_path(UPLOAD_ROOT, name).is_file():
        return {"name": name, "status": "unknown"}
    folder = storage.project_dir(UPLOAD_ROOT, name)
    return {
        "name": name,
        "status": "done",
        "meta": events.meta(folder),
        "tracks": events.summary(folder),
    }


def list_projects() -> dict[str, Any]:
    """Every project: its name, whether its ZIP is built, and meta.json (key, bpm)."""
    return {"projects": events.project_list(UPLOAD_ROOT)}


def get_events(
    name: str, track: str, start: float = 0.0, end: float | None = None, limit: int = 1000
) -> dict[str, Any]:
    """Events of one track as [time in seconds, label] rows with start <= time < end, at most
    limit rows (up to 10000). truncated says whether more rows matched."""
    _check_name(name)
    folder = storage.project_dir(UPLOAD_ROOT, name)
    tracks = events.tracks(folder)
    if track not in tracks:
        listed = ", ".join(tracks) or "none"
        raise ToolError(f"No track '{track}' in project '{name}'; tracks: {listed}")
    labels = events.labels(folder)
    limit = max(1, min(limit, MAX_ROWS))
    rows: list[list[Any]] = []
    truncated = False
    for time, label_id in events.rows(tracks[track]):
        if time < start or (end is not None and time >= end):
            continue
        if len(rows) == limit:
            truncated = True
            break
        rows.append([time, labels.get(label_id, str(label_id))])
    return {"name": name, "track": track, "rows": rows, "truncated": truncated}


def _start(name: str, work: Callable[[], None]) -> dict[str, Any]:
    try:
        job = jobs.start(name, work)
    except JobsFull as exc:
        raise ToolError(str(exc)) from None
    return {"name": name, "status": job.status}


def _check_name(name: str) -> None:
    if not storage.is_valid_name(name):
        raise ToolError(f"Invalid project name '{name[:100]}'")
