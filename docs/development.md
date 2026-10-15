# Development

## Setup

Python 3.14, ffmpeg and [Task](https://taskfile.dev/installation/) on PATH. Docker for the
container tasks.

```bash
task setup      # .venv with app and dev dependencies, git hooks, then task doctor
```

madmom builds from source; `task deps:setup` installs numpy and Cython first and the rest
with `--no-build-isolation`, which is what lets the build see them.

## Loop

```bash
task test K=youtube           # one area
task analyse FILE=clip.wav    # real madmom run
task run                      # dev server on :5000; task smoke in another shell
task lint:run                 # format
task validate                 # before committing - the same gates as CI and the hooks
```

## Tests

`tests/conftest.py` replaces `analysis.analyse` with a stub that writes one CSV, so the HTTP,
storage and URL tests run without madmom. `tests/test_analysis.py` covers the pure parts of
analysis (band classification, pitch routing, CSV format) with synthetic signals.

Not covered by tests: madmom itself, ffmpeg decoding and yt-dlp downloads. Changes there
are verified with `task analyse` and `task analyse:yt`.

## Gates

| Gate | Task |
| --- | --- |
| Task tree resolves | `task --list-all` |
| ruff lint and format, yamllint | `task lint:check` |
| pytest | `task test` |
| Documents name real paths; scripts documented | `task quality:docs:check` |
| No file past 300 lines | `task quality:long-files:check` |
| Compose file valid | `task docker:config` |

`task validate` runs all of them. CI ([.github/workflows/validate.yml](../.github/workflows/validate.yml))
runs the same on every push; the image build runs on manual dispatch only.

## Dependencies

1. Runtime pins live in `app/requirements.txt` only; the Dockerfile and `deps:setup` read
   them from there.
2. madmom is pinned to a commit ([ADR 0002](adr/0002-madmom-from-git.md)).
3. yt-dlp is unpinned ([ADR 0004](adr/0004-yt-dlp-unpinned.md)).

## Writing an ADR

See [adr/README.md](adr/README.md).
