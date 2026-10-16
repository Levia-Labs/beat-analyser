import json
import shutil
import sys
import wave

import numpy as np
import pytest

import analysis

HAS_BEATCORE = analysis.beatcore_available()


@pytest.mark.parametrize(
    ("pitch", "track"),
    [(21, "bass"), (47, "bass"), (48, "chords"), (71, "chords"), (72, "melody"), (108, "melody")],
)
def test_pitch_track_boundaries(pitch, track):
    assert analysis.pitch_track(pitch) == track


def test_write_tracks_format(tmp_path):
    analysis.write_tracks(tmp_path, {"kick": [(0.51234, "kick")], "snare": []})
    assert (tmp_path / "kick.csv").read_text() == "0.5123,128\n"
    assert (tmp_path / "snare.csv").read_text() == ""


def test_write_tracks_uses_label_ids(tmp_path):
    analysis.write_tracks(tmp_path, {"chords": [(1.0, "note_69")], "beats": [(2.0, "beat")]})
    assert (tmp_path / "chords.csv").read_text() == "1,69\n"
    assert (tmp_path / "beats.csv").read_text() == "2,131\n"


@pytest.mark.parametrize(
    ("time", "text"),
    [
        (0.0, "0"),
        (0.5, "0.5"),
        (0.51234, "0.5123"),
        (0.49999, "0.5"),
        (10.0, "10"),
        (123.4567, "123.4567"),
    ],
)
def test_timestamps_rounded_without_trailing_zeros(time, text):
    assert analysis.format_time(time) == text


def test_labels_file_decodes_every_id(tmp_path):
    analysis.write_labels(tmp_path)
    labels = json.loads((tmp_path / analysis.LABELS_NAME).read_text())
    assert labels["0"] == "note_0"
    assert labels["127"] == "note_127"
    assert labels["128"] == "kick"
    assert labels["129"] == "snare"
    assert labels["130"] == "hh"
    assert labels["131"] == "beat"
    assert len(labels) == 132


def test_percussion_tracks_split_by_label():
    result = {"onsets": [(0.0, "kick"), (0.25, "hh"), (0.5, "snare"), (1.0, "kick")]}
    tracks = analysis.percussion_tracks(result)
    assert tracks == {
        "kick": [(0.0, "kick"), (1.0, "kick")],
        "snare": [(0.5, "snare")],
        "hh": [(0.25, "hh")],
    }


def test_pitched_tracks_route_by_pitch():
    found = [(0.5, 1.0, 40, 0.6), (1.0, 2.0, 69, 0.5), (2.0, 2.5, 76, 0.4)]
    assert analysis.pitched_tracks(found) == {
        "bass": [(0.5, "note_40")],
        "chords": [(1.0, "note_69")],
        "melody": [(2.0, "note_76")],
    }


def test_rhythm_without_beatcore_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "beatcore", None)
    with pytest.raises(RuntimeError, match="beatcore is not installed"):
        analysis.rhythm(tmp_path / "x.wav")


@pytest.mark.skipif(
    not HAS_BEATCORE or shutil.which("ffmpeg") is None, reason="needs beatcore and ffmpeg"
)
def test_analyse_end_to_end(tmp_path):
    rate, bpm = 44100, 120.0
    t = np.arange(rate * 8) / rate
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-30 * (t % (60 / bpm)))
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) * ((t >= 2) & (t < 4))
    clip = tmp_path / "audio.wav"
    with wave.open(str(clip), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((0.3 * (kick + tone) * 32767).astype("<i2").tobytes())

    result = analysis.analyse(clip, tmp_path)

    assert result["bpm"] == pytest.approx(bpm, abs=1.5)
    for track in (*analysis.PERCUSSION_TRACKS, *analysis.PITCHED_TRACKS, "beats"):
        assert (tmp_path / f"{track}.csv").is_file()
    assert (tmp_path / analysis.LABELS_NAME).is_file()
    assert ",69\n" in (tmp_path / "chords.csv").read_text()
