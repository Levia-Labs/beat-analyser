# 0007. go-task is the interface for people and agents

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

Running this project involved remembering `docker compose` invocations, a venv path that
differs between Windows and Linux, and a madmom install order. Agents working on the
repository improvised the same commands repeatedly, each one a permission prompt and a
chance to get it wrong.

## Decision

Every operation is a task with a `desc`; `task --list` is the index and
[docs/tasks.md](../tasks.md) the explanation. Python scripts are reached through tasks.
Agents are allowed `task *` and denied the destructive ones by name in
`.claude/settings.json`.

## Consequences

**Cost.** go-task's shell is `mvdan.cc/sh`, not bash. `task` must be installed by hand.

**Benefit.** One allowlist entry covers routine work. Destructive tasks are gated twice:
`CONFIRM=yes` and a deny rule. The same commands run in CI, the git hooks and the Stop hook.

## Alternatives considered

**Makefile.** Poor on Windows, where this project is developed.

**Scripts under `scripts/`.** No index, no namespacing, no `requires:`.
