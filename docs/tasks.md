# Tasks

`task --list` is the authoritative index. This page groups the tasks by purpose and says
which are safe for an agent to run unprompted.

Safety column: **read** changes nothing; **local** writes inside the clone (`.venv`,
`uploads/`, formatting); **slow** takes minutes; **destructive** loses data and needs
`CONFIRM=yes`.

## Everyday

| Task | Does | Safety |
| --- | --- | --- |
| `task setup` | `.venv`, dependencies, git hooks, then `doctor` | local, slow first time |
| `task doctor` | Which tools and imports are present | read |
| `task test [K=expr]` | pytest, madmom stubbed | read |
| `task validate` | Every gate | read |
| `task analyse FILE=path` | Analyse one local file, print per-track summary. Runs in the image when `.venv` lacks beatcore | local |
| `task analyse:yt URL=...` | Download and analyse a YouTube video | local, network |
| `task summary NAME=song` | Events and time range per track | read |
| `task projects` | Projects and whether each has a ZIP | read |
| `task repair` | Reanalyse every project | local, slow |
| `task run [PORT=5000]` | Development server | local |
| `task smoke [URL=...]` | Probe a running server | read |

`ROOT=<dir>` on any of the project tasks points them at a different upload root.

## Namespaces

| Task | Does | Safety |
| --- | --- | --- |
| `deps:setup` | Install into `.venv`, beatcore too when a host `cargo` exists | local |
| `deps:beatcore` | Build beatcore into `.venv` (needs host Rust) | local, slow first time |
| `rust:test` | beatcore's tests in the pinned Rust image | read, slow first time |
| `rust:check`, `rust:fmt` | rustfmt and clippy / format in place | read / local |
| `rust:analyse FILE=` | beatcore-cli on a WAV: tempo, beats, labelled onsets | read |
| `eval:synthetic [SECONDS=]` | Tempo, beat and onset scores on generated clips, in the image | read |
| `eval:dir DIR=` | Tempo and beat scores on an annotated folder, in the image | read |
| `rust:lock` | Regenerate `beatcore/Cargo.lock` | local, network |
| `rust:host:test` | `cargo test` with a host toolchain | read |
| `dev:basic-pitch [VERSION=]` | Fetch and verify the basic-pitch model into `app/models/` | local, network |
| `deps:check` | `pip check` and outdated packages | read |
| `deps:yt-dlp` | Upgrade yt-dlp in `.venv` | local, network |
| `docker:build`, `docker:rebuild` | Build the image (cached / from scratch) | slow |
| `docker:up`, `docker:down` | Start (builds first) / stop the container | local |
| `docker:logs`, `docker:ps`, `docker:size` | Inspect | read |
| `docker:shell` | Shell in the running container | local |
| `docker:config` | Validate `docker-compose.yml` | read |
| `docker:e2e [PORT=8099]` | Build, start, upload a fixture, check refusals, stop | local, slow first time |
| `docker:hub:check` | Compare the Docker Hub page with `tasks/docker/hub.md` | read |
| `docker:hub:publish` | Publish `tasks/docker/hub.md` as the Docker Hub page. Uses the `docker login` credentials, or `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`. Denied to agents | publishes |
| `dev:fixture [NAME=] [SECONDS=] [BPM=]` | Synthetic clip with known events in `.cache/fixtures/` | local |
| `dev:e2e [URL=]` | Fixture upload and refusal checks against a running server | local |
| `dev:encoding`, `dev:encoding:fix` | Find / fix non-UTF-8 tracked text files | read / local |
| `dev:packages GITHUB= PYPI=` | Library stars, licence, versions, wheels | read, network |
| `git:status` | Uncommitted changes and upstream | read |
| `git:update` | Fetch `main`, merge into current branch | local, network |
| `git:commit [M=...] [SKIP_CHECKS=1]` | Validate, stage, pre-commit, commit, push | pushes, denied to agents |
| `lint:check`, `lint:run` | ruff and yamllint; `run` formats | read / local |
| `quality:check` | Doc-path and file-length gates | read |
| `quality:docs`, `quality:long-files:top` | The findings behind the gates | read |
| `quality:*:baseline` | Accept current findings ([ADR 0006](adr/0006-ratcheting-baselines.md)) | local |
| `precommit:setup`, `precommit:run` | Install / run the git hooks | local / read |
| `uploads:size` | Disk used | read |
| `uploads:fixtures:clean` | Delete `fixture-*` test projects only | local |
| `uploads:zips:clean` | Delete cached ZIPs; rebuilt on next request | local |
| `uploads:clean CONFIRM=yes` | Delete every project and its audio | destructive, denied to agents |
| `claude:budget`, `claude:resume` | Usage windows | read |
| `claude:skills:setup`, `claude:skills:verify` | caveman and superpowers plugins | local / read |
| `claude:history`, `claude:history:prompts`, `claude:history:inline` | Session command log: repeats, approval prompts, one-liners | read |
| `claude:allow:sync` | Write per-task allow rules into `.claude/settings.json` | person only, denied to agents |
| `claude:allow:check`, `claude:allow:report` | Missing per-task rules: fail / list only | read |
| `claude:hooks:check` | Run the Stop hook by hand | read |
| `graphify:update`, `graphify:query Q=` | Knowledge graph | local / read |

## Adding a task

1. Put it in the namespace folder it belongs to, or the root Taskfile if it is everyday use.
2. Give it a `desc`; with arguments, a `Usage:` line in the `desc` and `requires: vars:`.
3. Anything that deletes data gets a `preconditions` block testing `CONFIRM=yes`, and a
   deny rule in [.claude/settings.json](../.claude/settings.json).
4. Logic longer than a few lines of shell becomes a script under `tasks/`, named in that
   folder's README.
5. Add it to this page.
6. `Bash(task *)` already allows it. A person runs `task claude:allow:sync` to add the
   explicit rule; `task validate` lists tasks still missing one.
