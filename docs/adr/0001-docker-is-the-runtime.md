# 0001. Docker is the supported runtime; the host venv is for development

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

madmom is hard to install. The PyPI release predates NumPy 2 and current Python, it compiles
Cython extensions, and it needs ffmpeg and libsndfile at run time. Getting it to build on a
given machine has been the hardest part of this project.

## Decision

The container defined by `app/Dockerfile` is the supported way to run the service. A host
virtualenv (`task setup`) exists for the CLI, the tests and quick iteration, on a best-effort
basis.

## Consequences

**Cost.** The first image build compiles madmom and takes a long time. Two environments can
drift; `app/requirements.txt` is the single source of pins for both to limit that.

**Benefit.** One reproducible environment for running, documented in one file. The tests do
not need madmom at all, so CI stays fast.

## Alternatives considered

**Host install only.** What the README warned "barely works". Not reproducible.

**Conda.** Would make madmom's build easier on some machines and adds a second package
manager for everyone else.
