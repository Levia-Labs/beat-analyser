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
   │  storage.save_meta, storage.clear_tracks
   │  analysis.analyse(audio.wav, folder)       madmom, minutes per song
   ▼
storage.package ──► <root>/<name>_tracks.zip   written aside, then swapped in
```

## Modules

| Module | Knows about | Does not know about |
| --- | --- | --- |
| `app.py` | HTTP, Flask config, headers | madmom, file layout details |
| `cli.py` | argparse, printing | HTTP |
| `pipeline.py` | the build order, locking | where audio comes from |
| `analysis.py` | madmom, numpy, ffmpeg | folders other than the one it is given |
| `storage.py` | file layout, names, ZIPs | audio |
| `youtube.py` | URLs, yt-dlp | everything else |

`app.py` and `cli.py` are two front ends to `pipeline.py`; nothing in the pipeline imports
Flask.

## Analysis

1. **Onsets.** `CNNOnsetProcessor` produces activations at 100 frames per second;
   `OnsetPeakPickingProcessor(fps=100, threshold=0.03, combine=0.03)` turns them into onset
   times.
2. **Percussion class.** For each onset, an FFT over 25 ms either side, split into three
   bands: up to 150 Hz (kick), 150 Hz to 4 kHz (snare), above 4 kHz (hi-hat). The band with
   the largest weighted share of energy wins (weights 1.2, 1.1, 1.0).
3. **Notes.** `RNNPianoNoteProcessor` and `NotePeakPickingProcessor(threshold=0.15)` give
   onset time and MIDI pitch. Pitch routes the note to bass (<48), chords (48 to 71) or
   melody (72 and up).

The madmom processors are built once per process (`functools.cache`). Each loads network
weights from disk, which previously happened on every request.

madmom is imported inside the functions that use it, so `analysis` imports without madmom
and the tests run anywhere Flask and numpy do.

## Storage layout

See the Output section of the [README](../README.md). Project names are sanitised upload
stems or 11-character YouTube video IDs. `storage.is_valid_name` is the single check every
name passes before it becomes a path.

## Runtime

The container runs gunicorn with one worker and four threads: one worker so the models load
once, threads so a long analysis does not block the UI. The worker timeout is 30 minutes.
The image is built in two stages so the compiler and git used to build madmom do not ship.
