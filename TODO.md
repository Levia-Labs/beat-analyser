# TODO

Known gaps, roughly in priority order. An item here was identified and deliberately not done.

## Unverified

- [ ] No host Rust toolchain on the Windows development machine: `task deps:beatcore` and
      `task rust:host:test` are untested on a host; everything Rust ran in Docker.
- [ ] Bind-mounted `uploads/` owned by a uid other than 1000 on a Linux host
      (troubleshooting documents the `chown`).
- [ ] yt-dlp's need for a JavaScript runtime on current YouTube; the image ships none.

## Analysis quality

- [ ] **beatcore is unproven on real music** ([ADR 0009](docs/adr/0009-beatcore-and-basic-pitch.md)).
      `task eval:dir DIR=` scores an annotated folder; no dataset has been run yet
      (Ballroom: check size and licence first).
- [x] Drum classes, `task eval:synthetic` (36 clips, 2026-10-16), kick / snare / hh
      F-measure: raw sums 0.81 / 0.17 / 0.84; per-bin density (current) 0.86 / 0.83 /
      0.94, with no false snares on four-on-the-floor, white-noise hats, or the MP3 e2e
      fixture. Approaches tried and dropped are listed in `beatcore/src/onset.rs`.
- [ ] Tempo octave errors: every acc1 miss is at 70, 150 or 180 BPM (acc1 0.69, acc2
      1.00): the 120 BPM prior pulls toward the middle. A narrower prior trades one end
      for the other; tune on real data.
- [ ] `noisehat@110` beat F-measure 0.00 with the right tempo: the grid locked half a beat
      off, onto the hats. The beat envelope's hi-hat weight (0.35) is not low enough there.
- [ ] Kicks under a sustained pad: F-measure 0.08-0.48. The pad's low partials mask the
      kick's per-bin rise.
- [ ] One label per onset: a kick and hi-hat struck together count as one. Co-labelling
      was tried three ways and each added false labels (scores in `onset.rs`).
- [ ] `eval.py` scores WAV only; real uploads are mostly MP3. Score both.
- [ ] One global tempo; songs with tempo changes get one grid.
- [x] Key from the notes (`app/key.py`, Krumhansl-Schmuckler).
- [x] Re-triggered held notes are merged (`notes.merge_repeats`).
- [ ] beatcore holds the GIL while it runs (seconds per song). Releasing it needs PyO3's
      `detach`, whose name changed across 0.2x releases - pin first, then add.

## Behaviour

- [ ] Name collisions return a stale result ([ADR 0005](docs/adr/0005-name-keyed-projects.md)).
      Store a content hash in `meta.json` and reanalyse on mismatch.
- [ ] `/repair` and long analyses run inside the request. A job queue with a status endpoint
      would free the UI and survive a browser closing.

## Security

- [ ] Optional token authentication for a shared deployment
      ([ADR 0008](docs/adr/0008-loopback-without-auth.md)).
- [ ] No rate limiting.

## Tooling

- [ ] CI does not build the image on push (slow). A cached build job would catch Dockerfile
      breakage earlier.
- [ ] No dependency update automation (Dependabot or Renovate) for the pinned packages.

## 1 tasks/dev/gate.py raises UnicodeEncodeError on a failing gate under a cp1250 console (U+2716)

status: open
class: bug

## 2 dev:commit fails on a path whose deletion is already staged (git add: pathspec did not match); use git add -A -- <path> or skip add for staged deletions

status: open
class: bug
