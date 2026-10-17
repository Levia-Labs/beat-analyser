# Development

## Setup

Python 3.14, ffmpeg and [Task](https://taskfile.dev/installation/) on PATH. Docker for the
container and Rust tasks. A host Rust toolchain is optional.

```bash
task setup      # .venv with app and dev dependencies, git hooks, then task doctor
```

With `cargo` on PATH, `task deps:setup` also builds beatcore into `.venv` (`task deps:beatcore`
alone rebuilds it). Without, every Rust task runs in the pinned `rust` image and the analysis
tasks run in the app image.

## Loop

```bash
task test K=youtube           # one area of the Python tests
task rust:test                # beatcore's synthetic tempo, beat and onset tests
task analyse FILE=clip.wav    # a real run of both engines
task docker:e2e               # image build, upload, refusal checks
task run                      # dev server on :5000 (needs beatcore in .venv); task smoke
task lint:run                 # format Python; task rust:fmt for Rust
task validate                 # before committing - the same gates as CI and the hooks
task validate:deep            # before pushing - coverage, deep properties, security audits
```

## Tests

1. `tests/conftest.py` replaces `analysis.analyse` with a stub, so the HTTP, storage and URL
   tests run without beatcore or the note model.
2. `tests/test_notes.py` checks the ported note decoding on hand-built activations, and -
   with onnxruntime and ffmpeg present - transcribes synthetic tones of known pitch.
3. `tests/test_analysis.py` checks track routing, and runs both engines end to end when
   beatcore is built.
4. `beatcore/tests/synthetic.rs` holds drum loops and bursts with known tempo, beat times and
   classes.
5. `tests/test_properties.py` holds Hypothesis properties of the URL and name boundary, the
   time formatter and note merging: 50 examples in `task test`, 1000 in `task test:deep`.
   A failure prints the smallest falsifying input; Hypothesis replays it from `.hypothesis/`.

Not covered: analysis quality on real music (TODO.md), ffmpeg itself and yt-dlp downloads.

## Gates

| Gate | Task |
| --- | --- |
| Task tree resolves | `task --list-all` |
| ruff lint and format, yamllint | `task lint:check` |
| Types (basedpyright, baselined) | `task types:check` |
| Project conventions as ast-grep rules, and their tests | `task sg:check`, `task sg:test` |
| pytest, properties at 50 examples | `task test` |
| Documents name real paths; scripts documented | `task quality:docs:check` |
| No file past 300 lines | `task quality:long-files:check` |
| Compose file valid | `task docker:config` |
| Rust format, clippy, tests | `task rust:check`, `task rust:test` (CI job `rust`) |

Deep tier, `task validate:deep`, by hand before a push:

| Gate | Task |
| --- | --- |
| Branch coverage of `app/` never falls | `task test:cov`, `task quality:coverage:check` |
| Properties at 1000 examples | `task test:deep` |
| Known vulnerabilities in `.venv` | `task security:deps` (network) |
| Secrets in the git history | `task security:secrets`; staged changes by the pre-commit hook |
| RustSec advisories | `task rust:audit` (network) |

`task validate` runs the Python-side gates, offline. CI
([.github/workflows/validate.yml](../.github/workflows/validate.yml)) runs those and the Rust
job on every push; the image build runs on manual dispatch.

## Dependencies

1. Python runtime pins live in `app/requirements.txt` only.
2. Rust dependencies are locked in `beatcore/Cargo.lock`; `task rust:lock` regenerates it.
3. The note model is a committed file with a recorded sha256 (`app/models/README.md`);
   `task dev:basic-pitch` refreshes it.
4. yt-dlp is unpinned ([ADR 0004](adr/0004-yt-dlp-unpinned.md)).

## Writing an ADR

See [adr/README.md](adr/README.md).
