"""Audio to timed events, with madmom.

madmom is imported lazily and its processors are built once per process: each one loads
neural-network weights from disk, which used to happen on every request. Lazy import also
keeps this module importable where madmom is not installed (the CI test job).
"""

import csv
import subprocess
import warnings
from functools import cache
from pathlib import Path

import numpy as np

# CNNOnsetProcessor and RNNPianoNoteProcessor both emit activations at 100 frames per
# second. Peak picking must be told the same rate: the previous fps=250 scaled every onset
# time by 0.4, and combine=1 (seconds) merged all onsets inside one second into the first.
ACTIVATION_FPS = 100
ONSET_THRESHOLD = 0.03
ONSET_COMBINE_S = 0.03
NOTE_THRESHOLD = 0.15

PERCUSSION_TRACKS = ("kick", "snare", "hh")
PITCHED_TRACKS = ("bass", "chords", "melody")

# Classification window around each onset, and the band weights tuned by ear.
WINDOW_S = 0.025
BAND_WEIGHTS = {"kick": 1.2, "snare": 1.1, "hh": 1.0}

Event = tuple[float, str]

# madmom unpickles its models with an `align=0` NumPy 2.4 deprecates; harmless, and it
# otherwise fills the container log on every model load.
warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed")


@cache
def _onset_processors():
    from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor

    return CNNOnsetProcessor(), OnsetPeakPickingProcessor(
        fps=ACTIVATION_FPS, threshold=ONSET_THRESHOLD, combine=ONSET_COMBINE_S
    )


@cache
def _note_processors():
    from madmom.features.notes import NotePeakPickingProcessor, RNNPianoNoteProcessor

    return RNNPianoNoteProcessor(), NotePeakPickingProcessor(
        fps=ACTIVATION_FPS, threshold=NOTE_THRESHOLD
    )


def to_wav(source: Path, target: Path) -> None:
    """Decode anything ffmpeg reads into a PCM WAV."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-vn", str(target)],
        check=True,
        capture_output=True,
    )


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    from madmom.audio.signal import Signal

    signal = Signal(str(path), num_channels=1)
    return np.asarray(signal, dtype=np.float32), int(signal.sample_rate)


def classify_percussion(t: float, sr: int, y: np.ndarray) -> str | None:
    """kick, snare or hh by which frequency band dominates the window around t."""
    start = max(int((t - WINDOW_S) * sr), 0)
    end = min(int((t + WINDOW_S) * sr), len(y))
    frame = y[start:end]
    if len(frame) < 2:
        return None

    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1 / sr)
    total = spectrum.sum() + 1e-8
    bands = {
        "kick": spectrum[freqs <= 150].sum(),
        "snare": spectrum[(freqs > 150) & (freqs <= 4000)].sum(),
        "hh": spectrum[freqs > 4000].sum(),
    }
    return max(bands, key=lambda band: bands[band] / total * BAND_WEIGHTS[band])


def pitch_track(pitch: float) -> str:
    """MIDI pitch to track: bass below 48, chords 48-71, melody 72 and up."""
    if pitch < 48:
        return "bass"
    if pitch < 72:
        return "chords"
    return "melody"


def extract_percussion(audio_path: Path) -> dict[str, list[Event]]:
    detector, picker = _onset_processors()
    times = picker(detector(str(audio_path)))
    y, sr = load_mono(audio_path)

    tracks: dict[str, list[Event]] = {name: [] for name in PERCUSSION_TRACKS}
    for t in times:
        label = classify_percussion(float(t), sr, y)
        if label is not None:
            tracks[label].append((float(t), label))
    return tracks


def extract_pitched(audio_path: Path) -> dict[str, list[Event]]:
    detector, picker = _note_processors()
    notes = picker(detector(str(audio_path)))

    tracks: dict[str, list[Event]] = {name: [] for name in PITCHED_TRACKS}
    for onset, pitch in notes[:, :2]:
        tracks[pitch_track(pitch)].append((float(onset), f"note_{int(pitch)}"))
    return tracks


def write_tracks(folder: Path, tracks: dict[str, list[Event]]) -> None:
    for name, events in tracks.items():
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for time, label in events:
                writer.writerow([f"{time:.4f}", label])


def analyse(audio_path: Path, folder: Path) -> None:
    """Every track CSV for one audio file."""
    write_tracks(folder, extract_percussion(audio_path))
    write_tracks(folder, extract_pitched(audio_path))
