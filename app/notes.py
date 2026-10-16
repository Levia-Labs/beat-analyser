"""Polyphonic note transcription with Spotify's basic-pitch model, run through onnxruntime.

A port of basic-pitch 0.4.0 (`basic_pitch/inference.py`, `note_creation.py`), Copyright 2022
Spotify AB, Apache License 2.0 - see app/models/LICENSE-basic-pitch. Changes from upstream:

  - ONNX only. The TensorFlow, CoreML and TFLite branches are gone; upstream pins a
    TensorFlow that does not install on Python 3.14.
  - Audio is decoded and resampled to 22050 Hz mono by ffmpeg instead of librosa.
  - No MIDI output, no pitch bends, no frequency limits: this project writes CSV events.
  - librosa, resampy, pretty_midi and mir_eval are not needed.

The decoding algorithm (`_decode`) is upstream's `output_to_notes_polyphonic` unchanged.
"""

import subprocess
from functools import cache
from pathlib import Path

import numpy as np
from scipy.signal import argrelmax

MODEL_PATH = Path(__file__).resolve().parent / "models" / "nmp.onnx"

SAMPLE_RATE = 22050
FFT_HOP = 256
ANNOTATIONS_FPS = SAMPLE_RATE // FFT_HOP  # 86, integer division as upstream
WINDOW_SECONDS = 2
ANNOT_N_FRAMES = ANNOTATIONS_FPS * WINDOW_SECONDS
AUDIO_N_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS - FFT_HOP
N_OVERLAPPING_FRAMES = 30
OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN

MIDI_OFFSET = 21
MAX_FREQ_IDX = 87

# Upstream `predict` defaults.
ONSET_THRESHOLD = 0.5
FRAME_THRESHOLD = 0.3
MIN_NOTE_MS = 127.70
ENERGY_TOL = 11
# Gap below which two same-pitch notes are one held note (merge_repeats). About 4 frames.
MERGE_GAP_S = 0.05

Note = tuple[float, float, int, float]  # start s, end s, MIDI pitch, amplitude 0..1


@cache
def _session():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only
    return ort.InferenceSession(str(MODEL_PATH), options, providers=["CPUExecutionProvider"])


def load_audio(path: Path) -> np.ndarray:
    """Mono float32 at 22050 Hz, decoded by ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return np.frombuffer(result.stdout, dtype=np.float32)


def _windows(audio: np.ndarray):
    padded = np.concatenate([np.zeros(OVERLAP_LEN // 2, dtype=np.float32), audio])
    for start in range(0, padded.shape[0], HOP_SIZE):
        window = padded[start : start + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - len(window)))
        yield window.reshape(1, AUDIO_N_SAMPLES, 1)


def _unwrap(output: np.ndarray, original_length: int) -> np.ndarray:
    n_olap = N_OVERLAPPING_FRAMES // 2
    output = output[:, n_olap:-n_olap, :]
    frames = int(np.floor(original_length * (ANNOTATIONS_FPS / SAMPLE_RATE)))
    return output.reshape(-1, output.shape[2])[:frames, :]


def infer(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame and onset activations, each (n_frames, 88)."""
    session = _session()
    names = ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"]  # note, onset
    frames, onsets = [], []
    for window in _windows(audio):
        note, onset = session.run(names, {"serving_default_input_2:0": window})
        frames.append(note)
        onsets.append(onset)
    return (
        _unwrap(np.concatenate(frames), len(audio)),
        _unwrap(np.concatenate(onsets), len(audio)),
    )


def frame_times(n_frames: int) -> np.ndarray:
    """Upstream `model_frames_to_time`, magic alignment constant included."""
    index = np.arange(n_frames)
    window_numbers = np.floor(index / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / SAMPLE_RATE) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP)
    window_offset += 0.0018
    return index * FFT_HOP / SAMPLE_RATE - window_offset * window_numbers


def _inferred_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    diffs = []
    for n in range(1, n_diff + 1):
        appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(appended[n:, :] - appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    peak = np.max(frame_diff)
    if peak > 0:
        frame_diff = np.max(onsets) * frame_diff / peak
    return np.max([onsets, frame_diff], axis=0)


def _clear(energy: np.ndarray, rows, freq: int) -> None:
    energy[rows, freq] = 0
    if freq < MAX_FREQ_IDX:
        energy[rows, freq + 1] = 0
    if freq > 0:
        energy[rows, freq - 1] = 0


def _decode(frames: np.ndarray, onsets: np.ndarray, min_len: int) -> list[tuple]:
    """Upstream `output_to_notes_polyphonic` with infer_onsets and the melodia trick on."""
    n_frames = frames.shape[0]
    onsets = _inferred_onsets(onsets, frames)
    peaks = np.zeros(onsets.shape)
    relmax = argrelmax(onsets, axis=0)
    peaks[relmax] = onsets[relmax]
    times, freqs = np.where(peaks >= ONSET_THRESHOLD)
    energy = frames.copy()
    events = []

    for start, freq in zip(times[::-1], freqs[::-1], strict=True):
        if start >= n_frames - 1:
            continue
        i, k = start + 1, 0
        while i < n_frames - 1 and k < ENERGY_TOL:
            k = k + 1 if energy[i, freq] < FRAME_THRESHOLD else 0
            i += 1
        i -= k
        if i - start <= min_len:
            continue
        _clear(energy, slice(start, i), freq)
        events.append((start, i, freq + MIDI_OFFSET, float(np.mean(frames[start:i, freq]))))

    # Melodia trick: grow notes from the strongest remaining frames, both directions.
    while np.max(energy) > FRAME_THRESHOLD:
        mid, freq = np.unravel_index(np.argmax(energy), energy.shape)
        energy[mid, freq] = 0
        i, k = mid + 1, 0
        while i < n_frames - 1 and k < ENERGY_TOL:
            k = k + 1 if energy[i, freq] < FRAME_THRESHOLD else 0
            _clear(energy, i, freq)
            i += 1
        end = i - 1 - k
        i, k = mid - 1, 0
        while i > 0 and k < ENERGY_TOL:
            k = k + 1 if energy[i, freq] < FRAME_THRESHOLD else 0
            _clear(energy, i, freq)
            i -= 1
        begin = i + 1 + k
        if end - begin <= min_len:
            continue
        events.append((begin, end, freq + MIDI_OFFSET, float(np.mean(frames[begin:end, freq]))))
    return events


def merge_repeats(found: list[Note], gap: float = MERGE_GAP_S) -> list[Note]:
    """Join same-pitch notes separated by less than `gap` seconds.

    Not upstream. basic-pitch re-triggers a held note at every drum hit (an A4 held for one
    second came out as three notes), which turns one chord into several in chords.csv.
    The merged note keeps the first start, the last end and a duration-weighted amplitude.
    """
    merged: list[Note] = []
    open_by_pitch: dict[int, int] = {}
    for start, end, pitch, amp in sorted(found):
        i = open_by_pitch.get(pitch)
        if i is not None and start - merged[i][1] < gap:
            s0, e0, _, a0 = merged[i]
            d0, d1 = max(e0 - s0, 1e-6), max(end - start, 1e-6)
            merged[i] = (s0, max(e0, end), pitch, (a0 * d0 + amp * d1) / (d0 + d1))
            continue
        open_by_pitch[pitch] = len(merged)
        merged.append((start, end, pitch, amp))
    return sorted(merged)


def transcribe(path: Path) -> list[Note]:
    """Note events sorted by start time, repeated re-triggers merged."""
    frames, onsets = infer(load_audio(path))
    if frames.shape[0] == 0:
        return []
    min_len = int(np.round(MIN_NOTE_MS / 1000 * (SAMPLE_RATE / FFT_HOP)))
    times = frame_times(frames.shape[0])
    decoded = _decode(frames, onsets, min_len)
    return merge_repeats([(float(times[s]), float(times[e]), int(p), a) for s, e, p, a in decoded])
