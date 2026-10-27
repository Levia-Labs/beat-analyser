"""Reading what an analysis wrote: track CSVs, labels.json, meta.json.

Shared by the web app (app.py) and the MCP tools (mcp_tools.py), so it imports neither Flask
nor mcp. Readers, not validators: a malformed row is skipped, a missing or broken file reads
as empty. Callers check names with storage.is_valid_name before building a path.
"""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import storage

# analysis.LABELS_NAME. Not imported: analysis loads onnxruntime and the note model.
LABELS_NAME = "labels.json"
# Lane order in the viewer and in project_events: drums, beats, then the pitched tracks.
TRACK_ORDER = ("kick", "snare", "hh", "beats", "bass", "chords", "melody")


def tracks(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    found = {path.stem: path for path in folder.glob("*.csv")}
    ordered = [stem for stem in TRACK_ORDER if stem in found]
    ordered += sorted(stem for stem in found if stem not in TRACK_ORDER)
    return {stem: found[stem] for stem in ordered}


def rows(path: Path) -> Iterator[tuple[float, int]]:
    """(time, label ID) per CSV row."""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            try:
                yield float(row[0]), int(row[1])
            except IndexError, ValueError:
                continue


def labels(folder: Path) -> dict[int, str]:
    try:
        raw = json.loads((folder / LABELS_NAME).read_text(encoding="utf-8"))
        return {int(key): str(value) for key, value in raw.items()}
    except OSError, ValueError, AttributeError:
        return {}


def meta(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / storage.META_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None


def summary(folder: Path) -> list[dict[str, Any]]:
    result = []
    for track, path in tracks(folder).items():
        times = [time for time, _ in rows(path)]
        first, last = (times[0], times[-1]) if times else (None, None)
        result.append({"track": track, "events": len(times), "first": first, "last": last})
    return result


def project_list(root: Path) -> list[dict[str, Any]]:
    """Every project: name, whether its ZIP is built (the "done" rule), and meta.json."""
    if not root.is_dir():
        return []
    return [
        {
            "name": name,
            "done": storage.zip_path(root, name).is_file(),
            "meta": files.get(storage.META_NAME, {}).get("content"),
        }
        for name, files in storage.list_projects(root).items()
    ]


def project_events(root: Path, name: str) -> dict[str, Any]:
    """Every track's rows with the label table, for the viewer. name is already validated."""
    folder = storage.project_dir(root, name)
    return {
        "name": name,
        "meta": meta(folder),
        "labels": {str(key): value for key, value in labels(folder).items()},
        "tracks": {
            track: [[t, i] for t, i in rows(path)] for track, path in tracks(folder).items()
        },
    }
