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
      It passes synthetic tests with known truth only. Next: a `task eval` scoring tempo
      (within 4 percent, octave errors allowed) and beat F-measure with `mir_eval` on an
      annotated set (Ballroom: check size and licence first), with the old madmom results
      as the baseline to beat. Tune `PRIOR_BPM`, the peak-picking `delta` and
      `BEAT_WEIGHTS` there, not by ear.
- [ ] One label per onset: a kick and hi-hat struck together count as one event. Per-band
      peak picking with cross-band suppression would separate them.
- [ ] One global tempo; songs with tempo changes get one grid.
- [ ] `meta.json` carries a placeholder `key: C`. A key profile (Krumhansl) over the
      basic-pitch notes would fill it without another model.
- [ ] basic-pitch splits held notes at drum hits. Merging repeated same-pitch notes across a
      gap of a frame or two would clean chords.csv.
- [ ] beatcore holds the GIL while it runs (seconds per song). Releasing it needs PyO3's
      `detach`, whose name changed across 0.2x releases - pin first, then add.

## Behaviour

- [ ] Name collisions return a stale result ([ADR 0005](docs/adr/0005-name-keyed-projects.md)).
      Store a content hash in `meta.json` and reanalyse on mismatch.
- [ ] `/repair` and long analyses run inside the request. A job queue with a status endpoint
      would free the UI and survive a browser closing.
- [ ] No progress reporting during analysis.
- [ ] No way to delete one project from the UI.

## Security

- [ ] Optional token authentication for a shared deployment
      ([ADR 0008](docs/adr/0008-loopback-without-auth.md)).
- [ ] No rate limiting.
- [ ] Bootstrap from a CDN; vendoring it into `app/static/` would remove the external load.

## Tooling

- [ ] CI does not build the image on push (slow). A cached build job would catch Dockerfile
      breakage earlier.
- [ ] No dependency update automation (Dependabot or Renovate) for the pinned packages.
