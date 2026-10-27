"""One project from audio to ZIP. Shared by the HTTP routes and the command line.

A lock per project name serialises work on the same folder: two uploads of `song.mp3` at
once used to write the same CSVs concurrently. Different projects still run in parallel.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import analysis
import storage

log = logging.getLogger(__name__)

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(name: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def _ignore(stage: str) -> None:
    return None


def build(
    root: Path,
    name: str,
    place_audio: Callable[[Path], None],
    progress: Callable[[str], None] = _ignore,
) -> Path:
    """The project's ZIP, built if missing.

    place_audio(wav_path) writes the project's audio.wav; it is only called when the folder
    does not already hold one, so a failed analysis retries without re-downloading.
    """
    with _lock(name):
        cached = storage.zip_path(root, name)
        if cached.is_file():
            return cached

        folder = storage.project_dir(root, name)
        folder.mkdir(parents=True, exist_ok=True)
        wav = folder / storage.AUDIO_NAME
        if not wav.is_file():
            place_audio(wav)

        storage.clear_tracks(folder)
        result = analysis.analyse(wav, folder, progress)
        storage.save_meta(folder, name, key=result["key"], bpm=result["bpm"])
        log.info("built %s at %.2f BPM", name, result["bpm"])
        progress("package")
        return storage.package(root, name)


def repair_all(root: Path) -> list[dict]:
    """Regenerate CSVs, meta.json (with the detected tempo) and ZIP for every folder holding
    audio.wav."""
    repaired = []
    for folder in sorted(root.iterdir()):
        name = folder.name
        wav = folder / storage.AUDIO_NAME
        if not folder.is_dir() or not storage.is_valid_name(name) or not wav.is_file():
            continue
        try:
            with _lock(name):
                storage.clear_tracks(folder)
                result = analysis.analyse(wav, folder)
                storage.save_meta(folder, name, key=result["key"], bpm=result["bpm"])
                archive = storage.package(root, name)
            repaired.append({"folder": name, "zip": archive.name})
        except Exception as exc:  # one broken project must not stop the rest
            log.exception("repair failed for %s", name)
            repaired.append({"folder": name, "error": str(exc)})
    return repaired
