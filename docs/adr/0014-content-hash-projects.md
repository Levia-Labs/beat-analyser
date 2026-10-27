# 0014. A name collision compares the audio hash, and analyses are capped

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

[ADR 0005](0005-name-keyed-projects.md) served the cached ZIP for any upload with a known
name, so two different files called `song.wav` returned the first one's result. Separately,
nothing limited how many analyses ran at once: each decodes a full track and runs beatcore and
the note model, so a burst of uploads starved the host.

## Decision

`meta.json` records `audio_sha256`, the SHA-256 of the project's `audio.wav`. When an upload
meets a project with a ZIP, the new audio is written to `audio.new.wav` in the project folder
and hashed. The same hash serves the cached ZIP. A different hash clears the CSVs, ZIP and
meta, moves the new audio over `audio.wav` and analyses again, all under the per-project lock.
The old audio is replaced only after the new file is complete. A project without
`audio_sha256` counts as matching only when its `audio.wav` hashes the same.

At most `MAX_ANALYSES` (default 2) analyses run at once in the web app. A request that would
start another answers 429 and leaves the project as it was. A cached result takes no slot.

## Consequences

**Cost.** A repeat upload of a cached project is written to disk and hashed before it is
served, and a YouTube repeat downloads again to learn the hash.

**Cost.** A different file with the same name replaces the earlier result; the earlier one is
not kept. Names stay the key.

**Cost.** A request refused with 429 may already have written and removed a temporary file,
because the hash that decides whether it is cached needs the audio first.

**Benefit.** Collisions give the right result, and the host cannot be saturated by uploads.
Folder names remain readable.

## Alternatives considered

**Key by content hash.** Folder names become meaningless hashes and every consumer of the
folders changes. The hash in `meta.json` gives the same correctness.

**Queue excess analyses.** Holds an HTTP request open for minutes; a refusal the client can
retry is simpler.
