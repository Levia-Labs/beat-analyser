# 0002. madmom is installed from a pinned git commit

- **Status:** Superseded by [0009](0009-beatcore-and-basic-pitch.md) - madmom removed
- **Date:** 2026-10-15

## Context

The last madmom release on PyPI does not build against NumPy 2 or current Python. The
development branch on GitHub does, but it moves.

## Decision

Install madmom from commit `27f032e8947204902c675e5e341a3faf5dc86dae` on GitHub, in
`app/requirements.txt`. Install numpy and Cython first, then the requirements with
`--no-build-isolation`, so madmom's `setup.py` can import them.

## Consequences

**Cost.** A git clone at build time, so the build stage needs git and network access to
GitHub. Updating madmom is a deliberate edit and a full rebuild.

**Cost.** madmom prints a NumPy deprecation warning when it unpickles its models. Harmless,
filtered by the tasks.

**Benefit.** The build is reproducible and a moving upstream cannot break it.

## Alternatives considered

**PyPI madmom 0.16.1.** Does not build on this stack.

**Unpinned git dependency.** Any upstream commit could break the next build with no change
here to point at.

**Vendoring madmom.** A large tree with compiled extensions and model files to maintain.
