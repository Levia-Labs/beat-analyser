# Tasks

`task --list` is the authoritative index. This page groups the tasks by purpose and says
which are safe for an agent to run unprompted.

Safety column: **read** changes nothing; **local** writes inside the clone (`.venv`,
`uploads/`, formatting); **slow** takes minutes; **destructive** loses data and needs
`CONFIRM=yes`.

## Everyday

| Task                               | Does                                                                                                              | Safety                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `task setup`                       | `.venv`, dependencies, git hooks (`setup:hooks`), then `doctor`                                                   | local, slow first time |
| `setup:tools`, `setup:hooks`       | Shared: check the tools / install the pre-commit, pre-push and commit-msg hooks                                   | read / local           |
| `task doctor`                      | Which tools and imports are present                                                                               | read                   |
| `task test:run [-- -k expr]`       | pytest, madmom stubbed, prints only the verdict. Shared `test:failed`, `test:file F=`, `test:changed` rerun parts | read                   |
| `task validate`                    | Every offline gate; the Stop hook runs it, pre-commit and CI run its parts                                        | read                   |
| `task validate:deep`               | `validate`, coverage ratchet, deep properties, pip-audit, gitleaks, cargo-audit                                   | read, network          |
| `task coverage:run`                | Tests under branch coverage; prints only files below 100%                                                         | read                   |
| `task props:deep`                  | Property tests at 1000 examples                                                                                   | read                   |
| `task sync`                        | Fetch `main`, merge into current branch                                                                           | local, network         |
| `task analyse FILE=path`           | Analyse one local file, print per-track summary. Runs in the image when `.venv` lacks beatcore                    | local                  |
| `task analyse:yt URL=...`          | Download and analyse a YouTube video                                                                              | local, network         |
| `task summary NAME=song`           | Events and time range per track                                                                                   | read                   |
| `task projects`                    | Projects and whether each has a ZIP                                                                               | read                   |
| `task repair`                      | Reanalyse every project                                                                                           | local, slow            |
| `task run [PORT=5000]`             | Development server                                                                                                | local                  |
| `task smoke [URL=...]`             | Probe a running server                                                                                            | read                   |
| `task gate T="a,b K=x" [LINES=30]` | Tasks as gates: one line each, decisive failure lines, full log in `.cache` (gate folder)                         | as the tasks run       |
| `task outline F="path ..."`        | Functions, classes, headings or tasks with line numbers                                                           | read                   |

`ROOT=<dir>` on any of the project tasks points them at a different upload root.

## Namespaces

| Task                                                                | Does                                                                                                                                                    | Safety                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `deps:setup`                                                        | Install into `.venv`, beatcore too when a host `cargo` exists                                                                                           | local                          |
| `deps:beatcore`                                                     | Build beatcore into `.venv` (needs host Rust)                                                                                                           | local, slow first time         |
| `rust:test`                                                         | beatcore's tests in the pinned Rust image                                                                                                               | read, slow first time          |
| `rust:check`, `rust:fmt`                                            | rustfmt and clippy / format in place                                                                                                                    | read / local                   |
| `rust:analyse FILE=`                                                | beatcore-cli on a WAV: tempo, beats, labelled onsets                                                                                                    | read                           |
| `eval:synthetic [SECONDS=]`                                         | Tempo, beat and onset scores on generated clips, in the image                                                                                           | read                           |
| `eval:synthetic:diff [THRESHOLD=]`                                  | Means with deltas and only the clips that moved, against the baseline                                                                                   | read                           |
| `eval:synthetic:baseline`                                           | Accept the last diff run as the baseline (the `.cache` eval folder)                                                                                     | local                          |
| `eval:dir DIR=`                                                     | Tempo and beat scores on an annotated folder, in the image                                                                                              | read                           |
| `rust:lock`                                                         | Regenerate `beatcore/Cargo.lock`                                                                                                                        | local, network                 |
| `rust:host:test`                                                    | `cargo test` with a host toolchain                                                                                                                      | read                           |
| `dev:basic-pitch [VERSION=]`                                        | Fetch and verify the basic-pitch model into `app/models/`                                                                                               | local, network                 |
| `deps:check`                                                        | `pip check` and outdated packages                                                                                                                       | read                           |
| `deps:yt-dlp`                                                       | Upgrade yt-dlp in `.venv`                                                                                                                               | local, network                 |
| `docker:build`, `docker:rebuild`                                    | Build the image (cached / from scratch)                                                                                                                 | slow                           |
| `docker:up [PORT=80] [MCP_PORT=8001] [BIND=127.0.0.1]`              | Start the service and its MCP endpoint (builds first)                                                                                                   | local                          |
| `docker:down`                                                       | Stop and remove the container                                                                                                                           | local                          |
| `docker:logs`, `docker:ps`, `docker:size`                           | Inspect                                                                                                                                                 | read                           |
| `docker:shell`                                                      | Shell in the running container                                                                                                                          | local                          |
| `docker:config`                                                     | Validate `docker-compose.yml`                                                                                                                           | read                           |
| `docker:e2e [PORT=8099] [MCP_PORT=8098]`                            | Build, start, upload a fixture, check refusals, run `dev:mcp:e2e`, stop                                                                                 | local, slow first time         |
| `docker:hub:check`                                                  | Compare the Docker Hub page with `tasks/docker/hub.md`                                                                                                  | read                           |
| `docker:hub:publish`                                                | Publish `tasks/docker/hub.md` as the Docker Hub page. Uses the `docker login` credentials, or `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`. Denied to agents | publishes                      |
| `web:install`, `web:lock`                                           | `npm ci` from the lockfile / `npm install` after a `package.json` change, inside `web/`                                                                 | local, network                 |
| `web:dev`, `web:build`                                              | Vite dev server on :5173 proxied to `task run` / type-check and build `web/dist`                                                                        | local                          |
| `web:test`, `web:format`, `web:check`                               | vitest / prettier write / typecheck, eslint, prettier check and vitest (part of `validate`)                                                             | local                          |
| `dev:fixture [NAME=] [SECONDS=] [BPM=]`                             | Synthetic clip with known events in the `.cache` fixtures folder                                                                                        | local                          |
| `dev:e2e [URL=]`                                                    | Fixture upload and refusal checks against a running server                                                                                              | local                          |
| `dev:mcp:e2e [URL=] [TOKEN=]`                                       | Fixture analysis, status, events and a refusal through a running MCP service                                                                            | local                          |
| `dev:encoding`, `dev:encoding:fix`                                  | Find / fix non-UTF-8 tracked text files                                                                                                                 | read / local                   |
| `dev:packages GITHUB= PYPI=`                                        | Library stars, licence, versions, wheels                                                                                                                | read, network                  |
| `dev:pyapi -- module[:attr.path]`                                   | Signature and first docstring line of a Python attribute, or a module's public names, from the venv                                                     | read                           |
| `dev:commit [COAUTHOR=] -- "msg" path...`                           | Commit only the given paths; re-stage and retry once when hooks rewrote them. Never pushes                                                              | local                          |
| `dev:sdd:workspace PLAN=`, `dev:sdd:brief`, `dev:sdd:review`        | Superpowers plan workspace, task brief and review package (`PLAN= N=`, `PLAN= FROM= [TO=HEAD]`)                                                         | local                          |
| `git:status`, `git:head`, `git:log`, `git:files`                    | Shared: working tree and history, one line each                                                                                                         | read                           |
| `git:commit -- "message"`                                           | Shared: stage all, commit, push                                                                                                                         | pushes, denied to agents       |
| `lint:python:check`, `lint:yaml:check`                              | Shared lint over this repo's `ruff.toml` and `.yamllint`; `lint:python:run` formats                                                                     | read / local                   |
| `hostpy:check`, `hostpy:run`                                        | ruff over `tasks/` at its Python 3.12 target (`tasks/ruff.toml`); the shared lint cannot read that config                                               | read / local                   |
| `types:check`                                                       | basedpyright against `.basedpyright/baseline.json`                                                                                                      | read                           |
| `types:baseline`                                                    | Accept the current type errors ([ADR 0006](adr/0006-ratcheting-baselines.md))                                                                           | local                          |
| `sg:check`, `sg:test`                                               | ast-grep project rules / their tests (`tasks/sg/`)                                                                                                      | read                           |
| `sg:search P='...' [L=]`                                            | Structural code search; single-quote the pattern                                                                                                        | read                           |
| `audit:venv`                                                        | pip-audit over `.venv`                                                                                                                                  | read, network                  |
| `security:deps`, `security:secrets`                                 | Shared: advisory audit of lock and requirements files; gitleaks over the git history                                                                    | read, network                  |
| `rust:audit`                                                        | cargo-audit over `beatcore/Cargo.lock` in the Rust image                                                                                                | read, network, slow first time |
| `quality:gate`                                                      | Shared: file length, doc paths, Taskfile lint, encoding; baselines in `.ratchet/`                                                                       | read                           |
| `quality:docs`, `quality:long-files`                                | Shared: the findings behind the gates                                                                                                                   | read                           |
| `coverage:check`                                                    | Coverage of `app/` per file against its baseline (after `coverage:run`)                                                                                 | read                           |
| `coverage:baseline`                                                 | Accept current coverage ([ADR 0006](adr/0006-ratcheting-baselines.md))                                                                                  | local                          |
| `lint:precommit`, `lint:precommit:staged`                           | Shared: run the git hooks over all / staged files                                                                                                       | read                           |
| `uploads:size`                                                      | Disk used                                                                                                                                               | read                           |
| `uploads:fixtures:clean`                                            | Delete `fixture-*` test projects only                                                                                                                   | local                          |
| `uploads:zips:clean`                                                | Delete cached ZIPs; rebuilt on next request                                                                                                             | local                          |
| `uploads:clean CONFIRM=yes`                                         | Delete every project and its audio                                                                                                                      | destructive, denied to agents  |
| `claude:budget`, `claude:resume`                                    | Usage windows                                                                                                                                           | read                           |
| `claude:history`, `claude:history:prompts`, `claude:history:inline` | Session command log: repeats, approval prompts, one-liners                                                                                              | read                           |
| `claude:spend`, `claude:usage`                                      | Token spend per session and subagent (shared namespace)                                                                                                 | read                           |
| `claude:lanes`, `claude:lanes:reset`                                | Subagent lanes in use against the limit of 6 / forget them                                                                                              | read / local                   |
| `claude:allow:check`, `claude:allow:report`                         | Permission rules against the Taskfile interface: fail / list only                                                                                       | read                           |
| `claude:hooks:check`                                                | Run the Stop hook by hand                                                                                                                               | read                           |
| `graphify:update`, `graphify:query Q=`                              | Knowledge graph                                                                                                                                         | local / read                   |

## Adding a task

1. Put it in the namespace folder it belongs to, or the root Taskfile if it is everyday use.
2. Give it a `desc`; with arguments, a `Usage:` line in the `desc` and `requires: vars:`.
3. Anything that deletes data gets a `preconditions` block testing `CONFIRM=yes`, and a
   deny rule in [.claude/settings.json](../.claude/settings.json).
4. Logic longer than a few lines of shell becomes a script under `tasks/`, named in that
   folder's README.
5. Add it to this page.
6. `Bash(task *)` allows it; no per-task rule ([ADR 0011](adr/0011-tool-first-agent-loop.md)).
7. Its output is what the caller decides on: one line on success, the failing lines on
   failure, counts and deltas rather than tables. A full log goes to `.cache`.
