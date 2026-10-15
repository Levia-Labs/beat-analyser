"""Project folders under the upload root: naming, metadata, packaging, listing.

One project is one folder, `<root>/<name>/`, holding `audio.wav`, `meta.json` and one CSV per
track, plus a sibling `<root>/<name>_tracks.zip`. The ZIP is the cache: when it exists the
project is done and is served as is (`/repair` rebuilds it).

Every name that reaches the file system passes `is_valid_name` first. The name comes from an
uploaded filename or a URL path segment, so this is the path-traversal boundary.
"""

import json
import re
import zipfile
from pathlib import Path

from werkzeug.utils import secure_filename

AUDIO_NAME = "audio.wav"
META_NAME = "meta.json"
ZIP_SUFFIX = "_tracks.zip"

# No separators, no leading dot, bounded length. secure_filename already strips separators;
# this is the second check, applied to names that never went through it (URL segments).
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


def is_valid_name(name: str) -> bool:
    return bool(_NAME.fullmatch(name))


def project_name(filename: str) -> str:
    """The project name for an uploaded filename: its sanitised stem.

    Raises ValueError when nothing usable is left (`../../`, an all-unicode name).
    """
    # Stem first, then sanitise: secure_filename strips leading dots, so `.wav` would
    # otherwise become a project called `wav`.
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = secure_filename(base.rsplit(".", 1)[0] if "." in base else base)
    if not is_valid_name(stem):
        raise ValueError("File name has no usable characters")
    return stem


def project_dir(root: Path, name: str) -> Path:
    return root / name


def zip_path(root: Path, name: str) -> Path:
    return root / f"{name}{ZIP_SUFFIX}"


def save_meta(folder: Path, name: str, key: str = "C", bpm: float = 120.0) -> None:
    """Write meta.json. bpm is detected by beatcore; key is a placeholder (TODO.md)."""
    meta = {"name": name, "key": key, "bpm": bpm}
    (folder / META_NAME).write_text(json.dumps(meta), encoding="utf-8")


def clear_tracks(folder: Path) -> None:
    for csv_file in folder.glob("*.csv"):
        csv_file.unlink()


def package(root: Path, name: str) -> Path:
    """Zip the project folder under a `<name>/` prefix. Written aside, then swapped in, so a
    reader never receives a half-written archive."""
    folder = project_dir(root, name)
    target = zip_path(root, name)
    partial = target.with_suffix(".zip.part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{name}/{path.name}")
    partial.replace(target)
    return target


def list_projects(root: Path) -> dict[str, dict]:
    """Every project folder with its file sizes; meta.json content inlined."""
    result: dict[str, dict] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not is_valid_name(folder.name):
            continue
        files: dict[str, dict] = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            info: dict = {"size": path.stat().st_size}
            if path.name == META_NAME:
                try:
                    info["content"] = json.loads(path.read_text(encoding="utf-8"))
                except OSError, ValueError:
                    info["content"] = None
            files[path.name] = info
        result[folder.name] = files
    return result
