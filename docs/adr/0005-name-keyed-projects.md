# 0005. Projects are keyed by name and the ZIP is the cache

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

Analysis takes minutes. Uploading the same song twice, or requesting the same YouTube video,
should not run it twice.

## Decision

A project is named after the sanitised upload stem or the YouTube video ID. If
`<name>_tracks.zip` exists, it is returned without analysis. If `audio.wav` exists but the ZIP
does not (a failed run), the audio is reused and only analysis reruns. `/repair` and
`task repair` regenerate everything from the stored audio.

## Consequences

**Cost.** Two different files with the same name collide: the second upload gets the first
one's result. Documented in troubleshooting.

**Cost.** Changing the analysis does not invalidate old ZIPs; `task repair` does.

**Benefit.** Names are readable in the UI and on disk, and a project folder is a stable
artefact that other tools can consume.

## Alternatives considered

**Key by content hash.** Correct for collisions, and the folder names become meaningless
hashes. A hash could be added to `meta.json` and checked on upload; listed in TODO.md.
