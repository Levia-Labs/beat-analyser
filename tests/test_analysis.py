import numpy as np
import pytest

import analysis

SR = 44100


def tone(freq: float, seconds: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize(("freq", "label"), [(60, "kick"), (1000, "snare"), (9000, "hh")])
def test_classify_percussion_by_band(freq, label):
    assert analysis.classify_percussion(0.1, SR, tone(freq)) == label


def test_classify_percussion_outside_signal():
    assert analysis.classify_percussion(10.0, SR, tone(60)) is None


@pytest.mark.parametrize(
    ("pitch", "track"),
    [(21, "bass"), (47, "bass"), (48, "chords"), (71, "chords"), (72, "melody"), (108, "melody")],
)
def test_pitch_track_boundaries(pitch, track):
    assert analysis.pitch_track(pitch) == track


def test_write_tracks_format(tmp_path):
    analysis.write_tracks(tmp_path, {"kick": [(0.51234, "kick")], "snare": []})
    assert (tmp_path / "kick.csv").read_text() == "0.5123,kick\n"
    assert (tmp_path / "snare.csv").read_text() == ""
