# TODO

Known gaps, roughly in priority order. An item here was identified and deliberately not done.

## Unverified

- [ ] The rebuilt image (two-stage, non-root, gunicorn, read-only root) has not been built
      and run. `task docker:up`, then `task smoke URL=http://localhost`.
- [ ] `--no-build-isolation` for the whole requirements file in the Dockerfile: confirm no
      other package needs an isolated build.
- [ ] Bind-mounted `uploads/` owned by a uid other than 1000 on a Linux host
      (troubleshooting documents the `chown`).
- [ ] yt-dlp's need for a JavaScript runtime on current YouTube; the image ships none.

## Analysis quality

- [ ] `meta.json` carries placeholder `key: C` and `bpm: 120`. madmom has tempo
      (`TempoEstimationProcessor`) and key (`CNNKeyRecognitionProcessor`) models.
- [ ] Percussion classification is a three-band energy heuristic. A drum-transcription model
      would separate overlapping hits.
- [ ] Note detection is a piano model applied to everything.
- [ ] No regression test runs madmom. A short committed WAV and an expected-events test,
      skipped when madmom is absent, would catch parameter mistakes like the `fps=250` bug.

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
