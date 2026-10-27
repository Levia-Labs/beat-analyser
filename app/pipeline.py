"""One project from audio to ZIP. Shared by the HTTP routes and the command line.

A lock per project name serialises work on the same folder: two uploads of `song.mp3` at
once used to write the same CSVs concurrently. Different projects still run in parallel.
"""

import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import analysis
import storage

log = logging.getLogger(__name__)

# Beside audio.wav in the project folder, so the final move is a same-volume rename.
STAGED_NAME = "audio.new.wav"

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(name: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


class Busy(Exception):
    """A project is being analysed; its folder must not move or vanish under the analysis."""


@contextmanager
def try_lock(*names: str) -> Iterator[None]:
    """Hold every named project's lock, or raise Busy at once. Sorted, so two renames that
    name the same pair cannot deadlock."""
    held: list[threading.Lock] = []
    try:
        for name in sorted(set(names)):
            lock = _lock(name)
            if not lock.acquire(blocking=False):
                raise Busy(name)
            held.append(lock)
        yield
    finally:
        for lock in reversed(held):
            lock.release()


def delete(root: Path, name: str) -> None:
    with try_lock(name):
        storage.delete_project(root, name)


def rename(root: Path, old: str, new: str) -> None:
    with try_lock(old, new):
        if storage.project_dir(root, new).exists() or storage.zip_path(root, new).exists():
            raise FileExistsError(new)
        storage.rename_project(root, old, new)


def _ignore(stage: str) -> None:
    return None


class TooBusy(Exception):
    """Every analysis slot is taken; nothing was changed."""


def _place(folder: Path, place_audio: Callable[[Path], None]) -> Path:
    """Write the new audio beside the old, so the old stays until the new is complete."""
    staged = folder / STAGED_NAME
    try:
        place_audio(staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def build(
    root: Path,
    name: str,
    place_audio: Callable[[Path], None],
    progress: Callable[[str], None] = _ignore,
    slots: threading.BoundedSemaphore | None = None,
) -> Path:
    """The project's ZIP, built if missing or if the audio under this name changed.

    place_audio(wav_path) writes audio. With a ZIP present it always runs, into a temporary
    file, and the hash decides: same audio serves the ZIP, different audio replaces the
    project (ADR 0014). Without a ZIP it only runs when the folder holds no audio.wav, so a
    failed analysis retries without re-downloading.

    slots, when given, caps concurrent analyses: a full set raises TooBusy before anything
    is changed. A cached result never takes a slot.
    """
    with _lock(name):
        cached = storage.zip_path(root, name)
        folder = storage.project_dir(root, name)
        created = not folder.is_dir()
        wav = folder / storage.AUDIO_NAME
        staged: Path | None = None
        try:
            if cached.is_file() or not wav.is_file():
                folder.mkdir(parents=True, exist_ok=True)
                staged = _place(folder, place_audio)
                if cached.is_file() and storage.sha256_of(staged) == storage.audio_hash(folder):
                    staged.unlink()
                    return cached

            if slots is not None and not slots.acquire(blocking=False):
                raise TooBusy(name)
        except BaseException:
            if staged is not None:
                staged.unlink(missing_ok=True)
            if created:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        try:
            if staged is not None:
                cached.unlink(missing_ok=True)
                storage.clear_tracks(folder)
                (folder / storage.META_NAME).unlink(missing_ok=True)
                staged.replace(wav)
            else:
                storage.clear_tracks(folder)
            digest = storage.sha256_of(wav)
            result = analysis.analyse(wav, folder, progress)
            storage.save_meta(
                folder,
                name,
                key=result["key"],
                bpm=result["bpm"],
                audio_sha256=digest,
                duration=storage.wav_duration(wav),
            )
            log.info("built %s at %.2f BPM", name, result["bpm"])
            progress("package")
            return storage.package(root, name)
        finally:
            if slots is not None:
                slots.release()


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
                storage.save_meta(
                    folder,
                    name,
                    key=result["key"],
                    bpm=result["bpm"],
                    audio_sha256=storage.sha256_of(wav),
                    duration=storage.wav_duration(wav),
                )
                archive = storage.package(root, name)
            repaired.append({"folder": name, "zip": archive.name})
        except Exception as exc:  # one broken project must not stop the rest
            log.exception("repair failed for %s", name)
            repaired.append({"folder": name, "error": str(exc)})
    return repaired
