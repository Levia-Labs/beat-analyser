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
| `task analyse FILE=path` | Analyse one local file, print per-track summary | local, slow |
| `task analyse:yt URL=...` | Download and analyse a YouTube video | local, slow, network |
| `task summary NAME=song` | Events and time range per track | read |
| `task projects` | Projects and whether each has a ZIP | read |
| `task repair` | Reanalyse every project | local, slow |
| `task run [PORT=5000]` | Development server | local |
| `task smoke [URL=...]` | Probe a running server | read |

`ROOT=<dir>` on any of the project tasks points them at a different upload root.

## Namespaces

| Task | Does | Safety |
| --- | --- | --- |
| `deps:setup` | Install into `.venv` | local |
| `deps:check` | `pip check` and outdated packages | read |
| `deps:yt-dlp` | Upgrade yt-dlp in `.venv` | local, network |
| `docker:build`, `docker:rebuild` | Build the image (cached / from scratch) | slow |
| `docker:up`, `docker:down` | Start (builds first) / stop the container | local |
| `docker:logs`, `docker:ps`, `docker:size` | Inspect | read |
| `docker:shell` | Shell in the running container | local |
| `docker:config` | Validate `docker-compose.yml` | read |
| `git:status` | Uncommitted changes and upstream | read |
| `git:update` | Fetch `main`, merge into current branch | local, network |
| `git:commit [M=...] [SKIP_CHECKS=1]` | Validate, stage, pre-commit, commit, push | pushes, denied to agents |
| `lint:check`, `lint:run` | ruff and yamllint; `run` formats | read / local |
| `quality:check` | Doc-path and file-length gates | read |
| `quality:docs`, `quality:long-files:top` | The findings behind the gates | read |
| `quality:*:baseline` | Accept current findings ([ADR 0006](adr/0006-ratcheting-baselines.md)) | local |
| `precommit:setup`, `precommit:run` | Install / run the git hooks | local / read |
| `uploads:size` | Disk used | read |
| `uploads:zips:clean` | Delete cached ZIPs; rebuilt on next request | local |
| `uploads:clean CONFIRM=yes` | Delete every project and its audio | destructive, denied to agents |
| `claude:budget`, `claude:resume` | Usage windows | read |
| `claude:skills:setup`, `claude:skills:verify` | caveman and superpowers plugins | local / read |
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
