"""The Projects table's query: parse the query string, then filter, sort and page the projects.

Imports no Flask: app.py hands `request.args` to `parse` and turns QueryError into a 400.
Readers, not validators of the files: a project without a readable meta.json has unknown key,
bpm and duration, and those sort last in either order.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import events
import storage

SORTS = ("name", "key", "bpm", "duration", "created", "size")
ORDERS = ("asc", "desc")
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
KEYS = frozenset(
    f"{note} {mode}"
    for note in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    for mode in ("major", "minor")
)
MODES = ("major", "minor")


class QueryError(ValueError):
    """A query parameter the table cannot use; the message goes to the client."""


@dataclass(frozen=True)
class Query:
    q: str = ""
    keys: tuple[str, ...] = ()
    status: str = ""
    bpm_min: float | None = None
    bpm_max: float | None = None
    dur_min: float | None = None
    dur_max: float | None = None
    sort: str = "created"
    order: str = "desc"
    offset: int = 0
    limit: int = DEFAULT_LIMIT


def _number(args: Mapping[str, str], field: str) -> float | None:
    raw = args.get(field, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise QueryError(f"{field} must be a number") from None
    if not math.isfinite(value):
        raise QueryError(f"{field} must be a finite number")
    return value


def _integer(args: Mapping[str, str], field: str, default: int, low: int, high: int | None) -> int:
    raw = args.get(field, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise QueryError(f"{field} must be an integer") from None
    if value < low or (high is not None and value > high):
        bound = f"at least {low}" if high is None else f"between {low} and {high}"
        raise QueryError(f"{field} must be {bound}")
    return value


def _choice(args: Mapping[str, str], field: str, allowed: tuple[str, ...], default: str) -> str:
    value = args.get(field, "").strip() or default
    if value not in allowed:
        raise QueryError(f"{field} must be one of {', '.join(allowed)}")
    return value


def _keys(args: Mapping[str, str]) -> tuple[str, ...]:
    """A comma-separated list of key names or modes; empty entries are skipped."""
    keys: list[str] = []
    for part in args.get("key", "").split(","):
        key = part.strip()
        if not key:
            continue
        if key not in KEYS and key not in MODES:
            raise QueryError("key must be key names such as 'A minor', or major or minor")
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def parse(args: Mapping[str, str]) -> Query:
    """Validate the query string. Raises QueryError; never returns a Query outside its bounds."""
    status = args.get("status", "").strip()
    if status and status not in ("built", "pending"):
        raise QueryError("status must be built or pending")
    return Query(
        q=args.get("q", "").strip().lower(),
        keys=_keys(args),
        status=status,
        bpm_min=_number(args, "bpm_min"),
        bpm_max=_number(args, "bpm_max"),
        dur_min=_number(args, "dur_min"),
        dur_max=_number(args, "dur_max"),
        sort=_choice(args, "sort", SORTS, "created"),
        order=_choice(args, "order", ORDERS, "desc"),
        offset=_integer(args, "offset", 0, 0, None),
        limit=_integer(args, "limit", DEFAULT_LIMIT, 1, MAX_LIMIT),
    )


def _size(path: Path) -> int:
    # A build or delete can remove a file between listing and stat: it counts as 0 bytes.
    try:
        return path.stat().st_size if path.is_file() else 0
    except FileNotFoundError:
        return 0


def _folder_bytes(folder: Path) -> int | None:
    """Bytes in the folder's files, or None when the folder vanished mid-listing."""
    try:
        return sum(_size(path) for path in folder.iterdir())
    except FileNotFoundError:
        return None


def _created(meta_path: Path) -> str | None:
    try:
        stamp = meta_path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _item(root: Path, name: str) -> dict[str, Any] | None:
    folder = storage.project_dir(root, name)
    archive = storage.zip_path(root, name)
    meta = events.meta(folder) or {}
    key = meta.get("key")
    folder_size = _folder_bytes(folder) if folder.is_dir() else 0
    size = (folder_size or 0) + _size(archive)
    if folder_size is None and not archive.is_file():
        return None  # deleted while the page was being built
    return {
        "name": name,
        "done": archive.is_file(),
        "key": key if isinstance(key, str) and key != "unknown" else None,
        "bpm": _number_or_none(meta.get("bpm")),
        "duration": _number_or_none(meta.get("duration")),
        "created": _created(folder / storage.META_NAME),
        "size": size,
    }


def _matches(item: dict[str, Any], query: Query) -> bool:
    if query.q and query.q not in item["name"].lower():
        return False
    if query.keys:
        key = item["key"]
        if key is None or not any(k == key or key.endswith(f" {k}") for k in query.keys):
            return False
    if query.status and item["done"] != (query.status == "built"):
        return False
    return _within(item["bpm"], query.bpm_min, query.bpm_max) and _within(
        item["duration"], query.dur_min, query.dur_max
    )


def _within(value: float | None, low: float | None, high: float | None) -> bool:
    """Inclusive bounds; an unknown value fails any bound that is set."""
    if low is None and high is None:
        return True
    if value is None:
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def _names(root: Path) -> list[str]:
    found = {folder.name for folder in root.iterdir() if folder.is_dir()}
    found |= {
        path.name.removesuffix(storage.ZIP_SUFFIX) for path in root.glob(f"*{storage.ZIP_SUFFIX}")
    }
    return sorted(name for name in found if storage.is_valid_name(name))


def page(root: Path, query: Query) -> dict[str, Any]:
    """One page of the filtered, sorted projects, and the size of the filtered set."""
    found = [_item(root, name) for name in _names(root)] if root.is_dir() else []
    items = [item for item in found if item is not None and _matches(item, query)]
    field = query.sort
    descending = query.order == "desc"
    known = [item for item in items if item[field] is not None]
    unknown = [item for item in items if item[field] is None]

    def sort_value(item: dict[str, Any]) -> Any:
        value = item[field]
        return value.lower() if isinstance(value, str) else value

    known.sort(key=lambda item: (sort_value(item), item["name"]), reverse=descending)
    ordered = known + unknown
    return {"items": ordered[query.offset : query.offset + query.limit], "total": len(ordered)}
