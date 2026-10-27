"""Audio to timed events.

Rhythm comes from beatcore, this project's Rust extension (beatcore/): onsets classified as
kick, snare or hi-hat, the tempo, and beat positions. Pitched notes come from notes.py, a
port of Spotify's basic-pitch running on onnxruntime. Both replaced madmom (ADR 0009).

beatcore is imported lazily, so this module imports - and the tests run - where the Rust
extension is not built.
"""

import csv
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import key
import notes

PERCUSSION_TRACKS = ("kick", "snare", "hh")
PITCHED_TRACKS = ("bass", "chords", "melody")
BEAT_TRACK = "beats"

# CSVs carry a numeric label ID; labels.json decodes it. The table is fixed, not built per
# song, so one ID means the same thing in every project. A note's ID is its MIDI pitch.
LABELS_NAME = "labels.json"
LABELS: dict[int, str] = {pitch: f"note_{pitch}" for pitch in range(128)} | {
    128: "kick",
    129: "snare",
    130: "hh",
    131: "beat",
}
LABEL_IDS = {label: label_id for label_id, label in LABELS.items()}

Event = tuple[float, str]


def to_wav(source: Path, target: Path) -> None:
    """Decode anything ffmpeg reads into a PCM WAV."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-vn", str(target)],
        check=True,
        capture_output=True,
    )


def _beatcore():
    """The compiled extension, or None.

    `import beatcore` alone is not proof: from the repository root, the crate's source
    directory beatcore/ imports as an empty namespace package.
    """
    try:
        # Built into the image and into .venv only where a Rust toolchain exists.
        import beatcore  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return beatcore if hasattr(beatcore, "analyse") else None


def beatcore_available() -> bool:
    return _beatcore() is not None


def rhythm(audio_path: Path) -> dict:
    """beatcore's result: {"bpm", "confidence", "beats", "onsets": [(seconds, label)]}."""
    beatcore = _beatcore()
    if beatcore is None:
        raise RuntimeError(
            "beatcore is not installed. It is built into the Docker image; on the host it "
            "needs a Rust toolchain and task deps:setup (docs/development.md)."
        )
    return beatcore.analyse(str(audio_path))


def percussion_tracks(result: dict) -> dict[str, list[Event]]:
    tracks: dict[str, list[Event]] = {name: [] for name in PERCUSSION_TRACKS}
    for time, label in result["onsets"]:
        tracks[label].append((float(time), label))
    return tracks


def pitch_track(pitch: float) -> str:
    """MIDI pitch to track: bass below 48, chords 48-71, melody 72 and up."""
    if pitch < 48:
        return "bass"
    if pitch < 72:
        return "chords"
    return "melody"


def pitched_tracks(found: list[notes.Note]) -> dict[str, list[Event]]:
    tracks: dict[str, list[Event]] = {name: [] for name in PITCHED_TRACKS}
    for start, _end, pitch, _amplitude in found:
        tracks[pitch_track(pitch)].append((start, f"note_{pitch}"))
    return tracks


def format_time(seconds: float) -> str:
    """Seconds rounded to 4 decimals, trailing zeros dropped: 0.5, 1, 0.5123.

    Not `:g`: it counts significant digits, so 123.4567 would lose its last decimal.
    """
    return f"{seconds:.4f}".rstrip("0").rstrip(".")


def write_tracks(folder: Path, tracks: dict[str, list[Event]]) -> None:
    for name, events in tracks.items():
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for time, label in events:
                writer.writerow([format_time(time), LABEL_IDS[label]])


def write_labels(folder: Path) -> None:
    labels = {str(label_id): label for label_id, label in LABELS.items()}
    (folder / LABELS_NAME).write_text(json.dumps(labels), encoding="utf-8")


def _ignore(stage: str) -> None:
    return None


def analyse(audio_path: Path, folder: Path, progress: Callable[[str], None] = _ignore) -> dict:
    """Write every track CSV and labels.json for one audio file. Returns {"bpm", "confidence",
    "key"}. progress(stage) is called as each stage starts (app/progress.py)."""
    write_labels(folder)
    progress("rhythm")
    result = rhythm(audio_path)
    write_tracks(folder, percussion_tracks(result))
    write_tracks(folder, {BEAT_TRACK: [(float(t), "beat") for t in result["beats"]]})
    progress("notes")
    found = notes.transcribe(audio_path)
    write_tracks(folder, pitched_tracks(found))
    progress("key")
    return {
        "bpm": round(float(result["bpm"]), 2),
        "confidence": float(result["confidence"]),
        "key": key.estimate(found)[0],
    }
