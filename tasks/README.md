# tasks

One folder per Taskfile namespace. The root [Taskfile.yml](../Taskfile.yml) holds the
everyday tasks (`setup`, `analyse`, `run`, `sync`, `validate`); `task --list` is the index and
[docs/tasks.md](../docs/tasks.md) explains each.

| Folder                 | Namespace   | What it does                                                                                                             |
| ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------ |
| [deps/](deps/)         | `deps:`     | The `.venv` virtualenv: install, check, upgrade yt-dlp                                                                   |
| [docker/](docker/)     | `docker:`   | Build, start, stop, logs and checks for the container; `hub.py` and `hub.md` behind `docker:hub:*`, the Docker Hub page  |
| [rust/](rust/)         | `rust:`     | The beatcore crate: test, fmt, clippy, CLI - in the pinned Rust image                                                    |
| [eval/](eval/)         | `eval:`     | Score beatcore's tempo, beats and onsets against ground truth                                                            |
| [coverage/](coverage/) | `coverage:` | Per-file branch coverage ratchet of `app/`                                                                               |
| [types/](types/)       | `types:`    | basedpyright and its baseline                                                                                            |
| [sg/](sg/)             | `sg:`       | ast-grep: project rules, their tests, structural search                                                                  |
| [audit/](audit/)       | `audit:`    | pip-audit over `.venv`; `rust:audit` covers Cargo.lock                                                                   |
| [uploads/](uploads/)   | `uploads:`  | Runtime data: size, cached ZIPs, deletion behind `CONFIRM=yes`                                                           |
| [graphify/](graphify/) | `graphify:` | A knowledge graph over the repository                                                                                    |
| [web/](web/)           | `web:`      | The React SPA in `web/`: install, lock, dev server, build, test, format, check                                           |
| [dev/](dev/)           | `dev:`      | Fixtures, end-to-end checks, encoding, package lookups, the basic-pitch model fetch; `smoke.py` behind root `task smoke` |

`claude:`, `lint:`, `git:`, `quality:`, `security:`, `test:`, `setup:`, `repo:`, `todo:`, `graph:`, `ci:` and `rtk:` come from the Crepidoma submodule (`.crepidoma/taskfiles`), not from a folder here.

`ruff.toml` here targets Python 3.12 and is read by `task hostpy:check`: these scripts run on the host's `python3`, not the
3.14 container.
