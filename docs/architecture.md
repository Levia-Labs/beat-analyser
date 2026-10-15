# Architecture

## Request flow

```text
browser / CLI
   │
   ▼
app.py ── validate input ──► storage.project_name  or  youtube.video_id
   │
   ▼
pipeline.build(root, name, place_audio)        lock per project name
   │  ZIP exists? ──yes──► return it            (the ZIP is the cache)
   │  audio.wav missing? ─► place_audio(...)    save upload / ffmpeg decode / yt-dlp download
   │  storage.clear_tracks
   │  analysis.analyse(audio.wav, folder)       beatcore + basic-pitch, seconds per song
   │  storage.save_meta(bpm=detected)
   ▼
storage.package ──► <root>/<name>_tracks.zip   written aside, then swapped in
```

## Modules

| Module | Knows about | Does not know about |
| --- | --- | --- |
| `app.py` | HTTP, Flask config, headers | madmom, file layout details |
| `cli.py` | argparse, printing | HTTP |
| `pipeline.py` | the build order, locking | where audio comes from |
| `analysis.py` | which engine writes which track | folders other than the one it is given |
| `notes.py` | basic-pitch's model and decoding | tracks, files |
| `beatcore` (Rust) | DSP: STFT, onsets, tempo, beats | Python, files other than one WAV |
| `storage.py` | file layout, names, ZIPs | audio |
| `youtube.py` | URLs, yt-dlp | everything else |

`app.py` and `cli.py` are two front ends to `pipeline.py`; nothing in the pipeline imports
Flask.

## Analysis

Two engines, chosen in [ADR 0009](adr/0009-beatcore-and-basic-pitch.md).

**Rhythm - beatcore** (`beatcore/`, Rust, imported as a Python extension). One streaming STFT
at 100 frames per second (2048-point Hann window at 44.1 kHz) produces three things per frame:

1. **Onsets.** Log-magnitude spectral flux with a three-bin frequency max-filter on the
   reference frame (SuperFlux), peak-picked: a local maximum over 30 ms that rises 0.05 of
   the envelope's peak above its 100 ms / 70 ms moving average; onsets within 30 ms merge.
2. **Percussion class.** Positive linear-magnitude change per band - kick up to 150 Hz,
   snare 150 Hz to 4 kHz, hi-hat above - over the onset frame and the two after it. The band
   with the largest weighted rise wins (weights 1.2, 1.1, 1.0).
3. **Tempo and beats.** The band log-flux, each band normalised by its mean and weighted
   1.0 / 0.7 / 0.35 toward the low end, so off-beat hi-hats do not pull the grid. Tempo is
   the autocorrelation peak between 40 and 240 BPM under a log-normal prior centred on
   120 BPM, refined to sub-frame lag by parabolic interpolation. Beats come from Ellis's
   dynamic programming (tightness 100), weak beats trimmed at both ends.

**Notes - basic-pitch** (`app/notes.py`, ONNX model in `app/models/`). ffmpeg resamples to
22050 Hz mono; the model runs on 2-second windows with 30 frames of overlap; upstream's
decoding turns frame and onset activations into notes (onset threshold 0.5, frame threshold
0.3, minimum 128 ms). Pitch routes each note to bass (<48), chords (48 to 71) or melody
(72 and up).

The ONNX session is built once per process. beatcore is imported lazily, so `analysis`
imports - and the tests run - where the extension is not built.

## Storage layout

See the Output section of the [README](../README.md). Project names are sanitised upload
stems or 11-character YouTube video IDs. `storage.is_valid_name` is the single check every
name passes before it becomes a path.

## Runtime

The container runs gunicorn with one worker and four threads: one worker so the note model
loads once, threads so a long analysis does not block the UI. The worker timeout is 30
minutes. The image is built in three stages: the official `rust` image supplies the
toolchain, a Python stage builds the beatcore wheel and the virtualenv, and only the
virtualenv reaches the runtime stage.
