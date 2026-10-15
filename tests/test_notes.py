"""basic-pitch port. The decoding tests run anywhere; the model test needs onnxruntime and
ffmpeg and is skipped without them (the CI test job has neither)."""

import importlib.util
import shutil
import wave

import numpy as np
import pytest

import notes


def write_tones(path, tones, seconds=4.0, rate=44100):
    """tones: (start s, end s, frequency Hz)."""
    t = np.arange(int(seconds * rate)) / rate
    y = np.zeros_like(t)
    for start, end, freq in tones:
        y += 0.3 * np.sin(2 * np.pi * freq * t) * ((t >= start) & (t < end))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((y * 32767).astype("<i2").tobytes())


def test_decode_finds_one_held_note():
    frames = np.zeros((100, 88))
    onsets = np.zeros((100, 88))
    frames[20:60, 48] = 0.9  # MIDI 69
    onsets[20, 48] = 0.9
    events = notes._decode(frames, onsets, min_len=11)
    assert [(s, p) for s, _, p, _ in events] == [(20, 69)]
    _, end, _, amplitude = events[0]
    assert 58 <= end <= 60
    assert amplitude == pytest.approx(0.9)


def test_decode_drops_short_notes():
    frames = np.zeros((100, 88))
    onsets = np.zeros((100, 88))
    frames[20:25, 30] = 0.9
    onsets[20, 30] = 0.9
    assert notes._decode(frames, onsets, min_len=11) == []


def test_frame_times_start_at_zero_and_increase():
    times = notes.frame_times(500)
    assert times[0] == 0
    assert np.all(np.diff(times) > 0)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="needs ffmpeg and onnxruntime",
)
def test_transcribe_known_pitches(tmp_path):
    clip = tmp_path / "tones.wav"
    write_tones(clip, [(0.5, 1.5, 440.0), (1.5, 2.5, 523.25), (2.5, 3.5, 110.0)])
    found = notes.transcribe(clip)
    pitches = {p for _, _, p, _ in found}
    assert {69, 72, 45} <= pitches
    a4 = min((n for n in found if n[2] == 69), key=lambda n: n[0])
    assert a4[0] == pytest.approx(0.5, abs=0.06)
