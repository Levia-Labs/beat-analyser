# CLAUDE.md

Instructions for AI agents. Humans want [docs/development.md](docs/development.md).

A Flask service that turns audio (an upload or a YouTube URL) into timed percussion, beat and
note events: one CSV per track, packaged as a ZIP. Rhythm is beatcore, a Rust extension in
`beatcore/`; notes are basic-pitch on onnxruntime (`app/notes.py`). It runs in Docker; a host
virtualenv runs the CLI and the tests.

## The rules that matter most

1. **`task` is the interface.** `task --list` is the index; [docs/tasks.md](docs/tasks.md)
   explains each one. Prefer a task over an ad-hoc command, and write a task for a command
   you find yourself running twice.
2. **Do not run anything slow or destructive to "check it works".** `task repair` reanalyses
   every song. `task docker:rebuild` rebuilds every layer from scratch. `task uploads:clean`
   deletes data and is denied in [.claude/settings.json](.claude/settings.json), as are
   `task git:commit` (it pushes), `git push`, `git reset --hard`, `git clean` and `docker push`.
3. **`task validate` before saying done.** The Stop hook runs it after any edit to source,
   Taskfiles or docs and blocks the turn on failure.

## Tool-first loop and agents

Directives, agents and hooks come from the crepidoma plugin (github.com/Levia-Labs/Crepidoma,
submodule `.crepidoma/`; ADR 0011 is superseded by Crepidoma ADR-0001). Pillar-specific rules:

1. Lanes share the working tree: give each a disjoint set of paths, let each run scoped gates
   (`test:run -- -k <expr>`), and run `task validate` once after they finish. `task claude:lanes` lists lanes.
2. Gate through `task gate T="lint:python:run,test:run -- -k <area>,types:check,sg:check"`, never `validate`
   mid-lane. A change to `app/Dockerfile` or `app/requirements.txt` also needs
   `task gate T=docker:e2e`; say so in the receipt when it was not run.
3. `app/` changes: failing test first in `tests/`; the code conventions below hold. A new pure
   function at a trust boundary gets a property in `tests/test_properties.py`. Never touch
   `uploads/`, never regenerate a baseline to pass a gate.
4. New tool scripts go under `tasks/<namespace>/`, a test in `tests/test_<script>.py` loaded with
   `importlib.util.spec_from_file_location` as `tests/test_coverage_gate.py` does, a README row and
   a line in `docs/tasks.md`. New ast-grep rules go in `tasks/sg/rules` with a test in
   `tasks/sg/tests` and a row in `tasks/sg/README.md`.
5. Docs: a decision with a non-obvious reason gets an ADR from `docs/adr/template.md`, numbered
   next, with a row in `docs/adr/README.md`; verify with `task gate T="quality:docs"`.

Pillar agent: [beatcore-tuner](.claude/agents/beatcore-tuner.md) moves one beatcore metric,
judged by `task eval:synthetic:diff`. Shared agents (toolsmith, implementer, rule-smith,
docs-syncer, diff-reviewer) come from the plugin.

`cavecrew-investigator` locates code; `task graphify:query` usually answers first.

## Verifying a change

```bash
task gate T="rust:test,test:run -- -k notes"   # any tasks as gates: one line each, failures excerpted
task test:run                   # seconds; analysis stubbed, note model run if installed
task types:check                # basedpyright against its baseline
task sg:check                   # ast-grep project rules (tasks/sg/rules)
task validate                   # lint, types, ast-grep, tests, doc paths, file length, compose
task validate:deep              # + coverage ratchet, 1000-example properties, security audits
task rust:test                  # beatcore: synthetic tempo, beat and onset truth
task rust:check                 # rustfmt and clippy -D warnings
task dev:fixture                # clip with known events in .cache/fixtures/
task analyse FILE=.cache/fixtures/fixture-demo.wav   # both engines, per-track summary
task docker:e2e                 # build image, upload a fixture, check refusals, stop
```

A change under `beatcore/` is verified by `task rust:test` and `task rust:check`, then
`task docker:e2e` for the Python binding. A change to `app/notes.py` by `task test:run -- -k notes`.

A change to `app/Dockerfile` or `app/requirements.txt` is verified only by `task docker:e2e`.
If it was not run, say so. A dependency change also gets `task security:deps` (Python) or
`task rust:audit` (beatcore).

A new pure function at a trust boundary (names, URLs, parsing) gets a property in
`tests/test_properties.py`. A baseline (`types:baseline`, `coverage:baseline`) is
regenerated only to record progress, never to green a failure (ADR 0006, ADR 0010).

## Finding things cheaply

Every file read stays in context. Ask the local tools first, read second.

1. Relationships ("what calls X", "what depends on Y"): `task graphify:query Q="..."`. The
   Stop hook refreshes the graph after each validated turn.
2. Code shape ("every `subprocess.run` without `check=`"): `task sg:search P='...'`, single
   quotes, `$X` one node, `$$$X` many. Text grep misses calls split across lines.
3. Literal strings: the Grep tool.
4. A file's shape: `task outline F="app/app.py docs/tasks.md"`, functions and headings with
   line numbers.
5. Then Read, with `offset` and `limit` on the lines the search named.

Tasks print compact output already: ruff `concise`, pytest `-q --tb=short`, coverage lists
only files below 100%. RTK's global hook compresses raw shell commands but does not rewrite
`task`, so run the gates as `rtk task validate`, `rtk task test:run`, `rtk task types:check`:
[.rtk/filters.toml](.rtk/filters.toml) then collapses a clean run to `validate: ok` and keeps
every failing line. Trust is pinned to the file's hash: after editing it, a person runs
`rtk trust` again. `rtk gain` shows what it saved. Every task denied in settings.json is
denied with the `rtk` prefix too; a new deny rule needs both forms.

## Layout

| Path        | What it is                                                                           |
| ----------- | ------------------------------------------------------------------------------------ |
| `app/`      | The service. Modules listed in [app/README.md](app/README.md)                        |
| `beatcore/` | Rust: onsets, drum classes, tempo, beats ([beatcore/README.md](beatcore/README.md))  |
| `tests/`    | pytest. `conftest.py` stubs madmom out                                               |
| `tasks/`    | Taskfile namespaces and the scripts behind them ([tasks/README.md](tasks/README.md)) |
| `docs/`     | Architecture, API, configuration, security, troubleshooting, ADRs                    |
| `uploads/`  | Runtime data, gitignored. Never commit it, never delete it unasked                   |

## Models

1. Orchestration: Opus (`claude-opus-5-5`). Subagents: Sonnet (`claude-sonnet-5-5`), set as
   `CLAUDE_CODE_SUBAGENT_MODEL` in [.claude/settings.json](.claude/settings.json).
2. `task claude:budget` before a batch of agents. A PreToolUse hook refuses new agents at
   95 percent of any usage window, a catch-all agent type, and a seventh lane.

## Skills and plugins

Listed in [.claude/README.md](.claude/README.md). `superpowers` skills trigger on their own
descriptions (brainstorming, TDD, systematic debugging, verification). `caveman` compresses
replies only: code, commit messages, docs and security warnings stay in normal English. Use
`humanizer` on prose written for others (docs, ADRs, PR bodies).

## No approval prompts

An approval prompt stops the person's work. Every command must match the allowlist in
[.claude/settings.json](.claude/settings.json) on the first try.

0. **Bash tool only. Never PowerShell** - it is denied in settings.json.
1. **One command per call.** No `&&`, `;`, `|`, `$(...)`, `for` loops or multi-line scripts:
   a compound command is checked part by part and almost always prompts.
2. **No `cd`.** Every call starts in the repository root; tasks run there too.
3. **No inline `python -c`, no heredoc scripts, no `curl` to the internet, no `sed -i`.** If
   an operation is not a task yet, write a script under `tasks/`, put a task in front of
   it, name it in the folder's README, then run the task. It is then free in every later
   session.
4. Files: Read, Write, Edit, Grep, Glob. Never `cat`, `head` or `grep` through the shell.
5. Never edit `.claude/settings*.json` through the shell or a task: Claude Code always asks
   before a command changes its own permissions.
6. `task claude:history:prompts` lists every past command the allowlist did not cover. Read
   it at the start of a long session; turn each repeat into a task.

| Instead of                                          | Use                                                          |
| --------------------------------------------------- | ------------------------------------------------------------ |
| a `python -c` that writes a test WAV                | `task dev:fixture NAME=fixture-x`                            |
| `curl -F file=@... /upload`, checking a ZIP by hand | `task dev:e2e URL=...`                                       |
| `docker compose up`, sleep, curl, `down`            | `task docker:e2e`                                            |
| `curl api.github.com`, `pypi.org/pypi/.../json`     | `task dev:packages GITHUB="o/r" PYPI="pkg"`                  |
| `file`, `iconv` on a suspect file                   | `task dev:encoding`, `task dev:encoding:fix`                 |
| deleting a test project                             | name it `fixture-*`, then `task uploads:fixtures:clean`      |
| reading CSVs or listing `uploads/`                  | `task summary NAME=`, `task projects`                        |
| a grep sweep for "what calls X"                     | `task graphify:query Q=`                                     |
| a grep for a call shape                             | `task sg:search P='yt_dlp.YoutubeDL($$$A)'`                  |
| reading code to check a convention                  | a rule in `tasks/sg/rules/` with a test in `tasks/sg/tests/` |
| reading code for a None or type slip                | `task types:check`                                           |
| reading a whole file to find one function           | `task outline F=path`, then Read with offset                 |
| reading a gate's full output                        | `task gate T="docker:e2e"`                                   |
| reading the 36-row `eval:synthetic` table           | `task eval:synthetic:diff`                                   |

Do not re-read a file after editing it.

## Conventions

1. App modules import each other by bare name (`import storage`): gunicorn loads them from
   `/app`, and `pytest.ini` puts `app/` on the path.
2. Every name that reaches the file system goes through `storage.is_valid_name`. Every
   YouTube URL is reduced to a video ID by `youtube.video_id` before yt-dlp sees it.
3. No inline `<script>` or `style=""` in templates: the CSP forbids them.
4. Comments explain why. Comments recording what a previous version got wrong stay.
5. Decisions with non-obvious reasons get an ADR in [docs/adr/](docs/adr/README.md).
6. Shell inside Taskfiles is not bash (plugin directive `taskfile-shell`). Top-level vars of
   included Taskfiles share one scope: prefix them with the namespace. Use `{{.PY}}` for the venv
   interpreter and `{{.SYS_PY}}` for the system one. Pass root vars through the `includes:` entry.
7. A convention that can be checked structurally becomes an ast-grep rule with a test, not
   a sentence here. Conventions 1 and 3, and the yt-dlp half of 2, are enforced by
   `tasks/sg/rules/`.
8. Text I/O passes `encoding="utf-8"` (rule `text-io-encoding`).
