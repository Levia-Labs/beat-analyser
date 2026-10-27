"""Analysis progress the SPA polls while its POST is still open (GET /progress/<id>).

In memory and per process: the image runs one gunicorn worker (app/Dockerfile), so the thread
analysing and the thread answering a poll share this registry. A second worker would only see
its own entries (docs/adr/0013). The client picks the ID; it carries no authority, only a
place to look.
"""

import re
import threading
import time
from collections.abc import Callable
from typing import Any

UPLOAD_STAGES = ("decode", "rhythm", "notes", "key", "package")
YOUTUBE_STAGES = ("download", *UPLOAD_STAGES)
KEEP_SECONDS = 600

_ID = re.compile(r"[0-9a-f]{32}")
_entries: dict[str, dict[str, Any]] = {}
_guard = threading.Lock()


def is_valid_id(value: str | None) -> bool:
    return value is not None and _ID.fullmatch(value) is not None


class Tracker:
    """One request's progress. Without a valid ID every call is a no-op, so the routes need no
    branches for clients that do not poll (curl, the MCP jobs)."""

    def __init__(
        self,
        progress_id: str | None,
        stages: tuple[str, ...],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.id = progress_id if is_valid_id(progress_id) else None
        self.stages = stages
        self._clock = clock
        if self.id:
            with _guard:
                _prune(clock())
                _entries[self.id] = {
                    "stages": list(stages),
                    "stage": stages[0],
                    "done": 0,
                    "fraction": None,
                    "finished": False,
                    "error": None,
                    "started": clock(),
                    "ended": None,
                }

    def stage(self, name: str) -> None:
        if name in self.stages:
            self._update(stage=name, done=self.stages.index(name), fraction=None)

    def fraction(self, value: float) -> None:
        self._update(fraction=round(max(0.0, min(1.0, value)), 3))

    def finish(self, error: str | None = None) -> None:
        fields: dict[str, Any] = {"finished": True, "error": error, "ended": self._clock()}
        if error is None:
            fields["done"] = len(self.stages)
        self._update(**fields)

    def _update(self, **fields: Any) -> None:
        if not self.id:
            return
        with _guard:
            entry = _entries.get(self.id)
            if entry is not None:
                entry.update(fields)


def snapshot(progress_id: str, clock: Callable[[], float] = time.monotonic) -> dict | None:
    if not is_valid_id(progress_id):
        return None
    now = clock()
    with _guard:
        _prune(now)
        entry = _entries.get(progress_id)
        if entry is None:
            return None
        end = entry["ended"] if entry["ended"] is not None else now
        return {
            "stages": list(entry["stages"]),
            "stage": entry["stage"],
            "done": entry["done"],
            "total": len(entry["stages"]),
            "fraction": entry["fraction"],
            "finished": entry["finished"],
            "error": entry["error"],
            "elapsed": round(end - entry["started"], 1),
        }


def _prune(now: float) -> None:
    expired = [
        key
        for key, entry in _entries.items()
        if entry["ended"] is not None and now - entry["ended"] > KEEP_SECONDS
    ]
    for key in expired:
        del _entries[key]
