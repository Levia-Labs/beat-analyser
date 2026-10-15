"""Audio to timed events.

Rhythm comes from beatcore, this project's Rust extension (beatcore/): onsets classified as
kick, snare or hi-hat, the tempo, and beat positions. Pitched notes come from notes.py, a
port of Spotify's basic-pitch running on onnxruntime. Both replaced madmom (ADR 0009).

beatcore is imported lazily, so this module imports - and the tests run - where the Rust
extension is not built.
"""

import csv
import subprocess
from pathlib import Path

import notes

PERCUSSION_TRACKS = ("kick", "snare", "hh")
PITCHED_TRACKS = ("bass", "chords", "melody")
BEAT_TRACK = "beats"

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
        import beatcore
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


def write_tracks(folder: Path, tracks: dict[str, list[Event]]) -> None:
    for name, events in tracks.items():
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for time, label in events:
                writer.writerow([f"{time:.4f}", label])


def analyse(audio_path: Path, folder: Path) -> dict:
    """Write every track CSV for one audio file. Returns {"bpm", "confidence"}."""
    result = rhythm(audio_path)
    write_tracks(folder, percussion_tracks(result))
    write_tracks(folder, {BEAT_TRACK: [(float(t), "beat") for t in result["beats"]]})
    write_tracks(folder, pitched_tracks(notes.transcribe(audio_path)))
    return {"bpm": round(float(result["bpm"]), 2), "confidence": float(result["confidence"])}
