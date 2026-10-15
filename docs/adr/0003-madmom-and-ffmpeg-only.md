# 0003. madmom and ffmpeg do all audio I/O; librosa and pydub removed

- **Status:** Superseded by [0009](0009-beatcore-and-basic-pitch.md) - ffmpeg still decodes everything; madmom is gone, and librosa and pydub stay out
- **Date:** 2026-10-15

## Context

The code used three libraries to read audio. madmom read the file for detection, librosa
read it again for percussion classification, and pydub converted YouTube downloads to WAV.
librosa was installed in the Dockerfile but missing from `requirements.txt`, so the host
venv could not run the app, and it pulls in numba. pydub relies on `audioop`, which Python
3.13 removed, and falls back to a pure-Python version.

## Decision

`madmom.audio.signal.Signal` loads audio for classification. Decoding to WAV is one direct
`ffmpeg` call (`analysis.to_wav`). librosa and pydub are gone.

## Consequences

**Cost.** ffmpeg must be on PATH for any non-WAV input, on the host as well as in the image.
It already had to be: madmom uses it for compressed formats.

**Benefit.** Two fewer dependencies, numba among them. One decoder for every source: uploads
other than WAV are now converted too, where previously an MP3 was saved under the name
`audio.wav` and left to whichever library read it.

## Alternatives considered

**Keep librosa.** Its one use was `librosa.load(path, sr=None, mono=True)`, which madmom's
`Signal` does with what is already installed.

**soundfile.** Reads WAV and FLAC, not MP3 or M4A, so ffmpeg would still be needed.
