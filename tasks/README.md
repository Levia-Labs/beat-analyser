# tasks

One folder per Taskfile namespace. The root [Taskfile.yml](../Taskfile.yml) holds the
everyday tasks (`setup`, `test`, `analyse`, `run`, `validate`); `task --list` is the index and
[docs/tasks.md](../docs/tasks.md) explains each.

| Folder | Namespace | What it does |
| --- | --- | --- |
| [deps/](deps/) | `deps:` | The `.venv` virtualenv: install, check, upgrade yt-dlp |
| [docker/](docker/) | `docker:` | Build, start, stop, logs and checks for the container |
| [rust/](rust/) | `rust:` | The beatcore crate: test, fmt, clippy, CLI - in the pinned Rust image |
| [eval/](eval/) | `eval:` | Score beatcore's tempo, beats and onsets against ground truth |
| [git/](git/) | `git:` | `commit`, `status`, `update` - same commands as the toolkit repository |
| [lint/](lint/) | `lint:` | ruff and yamllint |
| [quality/](quality/) | `quality:` | Documentation-path and file-length gates |
| [precommit/](precommit/) | `precommit:` | The git hooks |
| [uploads/](uploads/) | `uploads:` | Runtime data: size, cached ZIPs, deletion behind `CONFIRM=yes` |
| [claude/](claude/) | `claude:` | Usage budget, hooks, caveman and superpowers plugins |
| [graphify/](graphify/) | `graphify:` | A knowledge graph over the repository |
| [dev/](dev/) | `dev:` | Fixtures, end-to-end checks, encoding, package lookups, the basic-pitch model fetch; `smoke.py` behind root `task smoke` |

`ruff.toml` here targets Python 3.12: these scripts run on the host's `python3`, not the
3.14 container.
