# CLAUDE.md

Instructions for AI agents. Humans want [docs/development.md](docs/development.md).

A Flask service that turns audio (an upload or a YouTube URL) into timed percussion and note
events with madmom: one CSV per track, packaged as a ZIP. It runs in Docker; a host
virtualenv runs the CLI and the tests.

## The rules that matter most

1. **`task` is the interface.** `task --list` is the index; [docs/tasks.md](docs/tasks.md)
   explains each one. Prefer a task over an ad-hoc command, and write a task for a command
   you find yourself running twice.
2. **Do not run anything slow or destructive to "check it works".** `task repair` reanalyses
   every song (minutes each). `task docker:rebuild` recompiles madmom. `task uploads:clean`
   deletes data and is denied in [.claude/settings.json](.claude/settings.json), as are
   `task git:commit` (it pushes), `git push`, `git reset --hard`, `git clean` and `docker push`.
3. **`task validate` before saying done.** The Stop hook runs it after any edit to source,
   Taskfiles or docs and blocks the turn on failure.

## Verifying a change

```bash
task test                       # seconds, no madmom needed: analysis is stubbed
task validate                   # lint, tests, doc paths, file length, compose file
task analyse FILE=song.wav      # real madmom run on one file, prints a per-track summary
task summary NAME=song          # what an analysed project contains, without reading CSVs
task run   then   task smoke    # web UI on :5000, then probe its headers and /folders
```

A change to `app/analysis.py` is not verified by `task test` alone: the tests cover band
classification and pitch routing, not madmom. Run `task analyse` on a short real file.

A change to `app/Dockerfile` or `app/requirements.txt` is verified only by `task docker:build`
(slow on first build). If it was not run, say so.

## Layout

| Path | What it is |
| --- | --- |
| `app/` | The service. Modules listed in [app/README.md](app/README.md) |
| `tests/` | pytest. `conftest.py` stubs madmom out |
| `tasks/` | Taskfile namespaces and the scripts behind them ([tasks/README.md](tasks/README.md)) |
| `docs/` | Architecture, API, configuration, security, troubleshooting, ADRs |
| `uploads/` | Runtime data, gitignored. Never commit it, never delete it unasked |

## Models

1. Orchestration: Opus (`claude-opus-5-5`). Subagents: Sonnet (`claude-sonnet-5-5`), set as
   `CLAUDE_CODE_SUBAGENT_MODEL` in [.claude/settings.json](.claude/settings.json).
2. `task claude:budget` before a batch of agents. A PreToolUse hook refuses new agents at
   95 percent of any usage window.

## Skills and plugins

Listed in [.claude/README.md](.claude/README.md). `superpowers` skills trigger on their own
descriptions (brainstorming, TDD, systematic debugging, verification). `caveman` compresses
replies only: code, commit messages, docs and security warnings stay in normal English. Use
`humanizer` on prose written for others (docs, ADRs, PR bodies).

## No approval prompts

An approval prompt stops the person's work. Every command must match the allowlist in
[.claude/settings.json](.claude/settings.json) on the first try.

1. **One command per call.** No `&&`, `;`, `|`, `$(...)`, `for` loops or multi-line scripts:
   a compound command is checked part by part and almost always prompts.
2. **No `cd`.** Every call starts in the repository root; tasks run there too.
3. **No inline `python -c`, no heredoc scripts, no `curl` to the internet, no `sed -i`.** If
   an operation is not a task yet, write a script under `tasks/`, put a task in front of
   it, name it in the folder's README, then run the task. It is then free in every later
   session.
4. Files: Read, Write, Edit, Grep, Glob. Never `cat`, `head` or `grep` through the shell.
5. Never edit `.claude/settings*.json` through the shell or a task: Claude Code always asks
   before a command changes its own permissions. `task claude:allow:sync` is for a person.
6. `task claude:history:prompts` lists every past command the allowlist did not cover. Read
   it at the start of a long session; turn each repeat into a task.

| Instead of | Use |
| --- | --- |
| a `python -c` that writes a test WAV | `task dev:fixture NAME=fixture-x` |
| `curl -F file=@... /upload`, checking a ZIP by hand | `task dev:e2e URL=...` |
| `docker compose up`, sleep, curl, `down` | `task docker:e2e` |
| `curl api.github.com`, `pypi.org/pypi/.../json` | `task dev:packages GITHUB="o/r" PYPI="pkg"` |
| `file`, `iconv` on a suspect file | `task dev:encoding`, `task dev:encoding:fix` |
| deleting a test project | name it `fixture-*`, then `task uploads:fixtures:clean` |
| reading CSVs or listing `uploads/` | `task summary NAME=`, `task projects` |
| a grep sweep for "what calls X" | `task graphify:query Q=` |

Do not re-read a file after editing it.

## Conventions

1. App modules import each other by bare name (`import storage`): gunicorn loads them from
   `/app`, and `pytest.ini` puts `app/` on the path.
2. Every name that reaches the file system goes through `storage.is_valid_name`. Every
   YouTube URL is reduced to a video ID by `youtube.video_id` before yt-dlp sees it.
3. No inline `<script>` or `style=""` in templates: the CSP forbids them.
4. Comments explain why. Comments recording what a previous version got wrong stay.
5. Decisions with non-obvious reasons get an ADR in [docs/adr/](docs/adr/README.md).
6. Shell inside Taskfiles is not bash: [.claude/rules/taskfile-shell.md](.claude/rules/taskfile-shell.md).
