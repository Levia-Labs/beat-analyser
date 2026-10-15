# 0009. beatcore (Rust) and basic-pitch (ONNX) replace madmom

- **Status:** Accepted
- **Date:** 2026-10-15
- **Supersedes:** [0002](0002-madmom-from-git.md), [0003](0003-madmom-and-ffmpeg-only.md)

## Context

madmom was the hardest dependency in the project: last PyPI release in 2018, installable on
Python 3.14 only from a pinned git commit, Cython extensions compiled on every image build,
and a NumPy deprecation warning on every model load. Its note model is a piano model applied
to every instrument. Its CNN onset detector was good, and was used with wrong parameters for
most of this project's life (`fps=250, combine=1`), which nobody noticed because nothing
measured it.

The replacements surveyed (October 2026, star counts from the GitHub API):

| Candidate | Stars | Why not |
| --- | --- | --- |
| librosa | 8.7k | Works on 3.14, but numba and llvmlite add ~150 MB and an 18 s first-call JIT, for one onset list and a tempo |
| essentia | 3.8k | AGPL-3.0 |
| aubio | 3.8k | GPL-3.0; last PyPI release 2019, no wheels |
| beat_this, BeatNet, all-in-one | < 1k | PyTorch (gigabytes) |
| Rust crates | < 300 | Nothing for tempo, beats or onsets near the bar |

For notes, Spotify's basic-pitch (5.9k stars, Apache-2.0) is instrument-agnostic, gives note
ends and amplitudes, and ships a 230 KB ONNX model. Its PyPI package pins a TensorFlow that
cannot install on Python 3.14, and its `resampy` pin imports the removed `pkg_resources`.

## Decision

1. **Rhythm: beatcore**, a Rust extension in `beatcore/` (PyO3, abi3 wheel built by maturin).
   One STFT pass at 100 frames per second gives a SuperFlux onset envelope (onsets), a
   low-weighted three-band envelope (tempo by autocorrelation with a log-normal prior at
   120 BPM; beats by Ellis's dynamic programming), and per-band flux (kick / snare / hi-hat).
2. **Notes: `app/notes.py`**, a port of basic-pitch's ONNX inference and note decoding, run by
   onnxruntime, with ffmpeg resampling. The model file is committed (`app/models/nmp.onnx`),
   fetched and sha256-verified from the PyPI wheel by `task dev:basic-pitch`.
3. The detected tempo goes into `meta.json`; beat times into a new `beats.csv`.

## Consequences

**Cost.** Custom DSP is code this project maintains, and on real music it is not proven
better than madmom's neural onset detector. It is tested on synthetic signals with known
truth (`task rust:test`); an evaluation on annotated music is still to do (TODO.md).

**Cost.** Building the extension needs Rust. The image build takes the toolchain from the
official `rust` image in a build stage; on a host without Rust, `task deps:setup` skips it
and the analysis tasks run in the image instead.

**Cost.** `app/notes.py` is a port, so upstream fixes have to be carried over by hand.

**Benefit.** Python 3.14 native, no git dependency, no Cython, no NumPy warning. A tempo and
beat grid exist for the first time. Notes for every instrument, not only piano.

**Benefit.** The image dropped madmom, Cython, mido and libsndfile; onnxruntime is the
largest addition.

## Alternatives considered

**librosa for rhythm.** Proven and well documented. Rejected for its weight and JIT start-up
on a path that needs one envelope, and because its single onset list would still need the
band classifier this project already had.

**Keep madmom for onsets, basic-pitch for notes.** Keeps the strongest onset detector and
all of madmom's install pain. The pain was the reason for this change.

**Pure-numpy DSP instead of Rust.** Would avoid the toolchain. Chosen against at the
person's request for a compiled core; the streaming STFT and the O(n * period) beat search
are where compiled code pays.
