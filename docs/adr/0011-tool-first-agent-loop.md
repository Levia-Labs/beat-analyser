# 0011. Agents decide which tool to run; tools return only what the decision needs

- **Status:** Accepted
- **Date:** 2026-10-20

## Context

`task claude:tokens` over the first six sessions: 113 million tokens, 98 percent of them
cache reads. A turn re-reads the whole context, so cost grows with what has been pasted into
it, not with the work done. 775k characters came back from Read alone; `.claude/settings.json`
was read 7 times (74k characters, most of it 170 per-task allow rules that `Bash(task *)`
already covered and 147 rules for the denied PowerShell tool); `Taskfile.yml`, `hooks.py` and `TODO.md` 6 to 8 times each; whole gate
outputs (cargo builds, image builds, the 36-row eval table) were read on every run. The one
reviewer subagent spent 968k tokens to return 8.6k characters.

The waste has one shape: a model reads content to produce a one-off answer that a
parametrized tool could have produced in a line.

## Decision

A model turn is a tool call with chosen parameters, and a tool's output is the minimum the
next decision needs. When no tool fits, one is written first (script, test, task) and then
called. Subagents are named single-job agents in `.claude/agents/` that call tasks and return
a receipt of at most 10 lines; catch-all agent types are refused by a hook, and at most six
lanes run at once.

Mechanisms:

1. `tasks/claude/lanes.py`, called by `hooks.py` on PreToolUse(Agent), SubagentStart and
   SubagentStop: refuses general-purpose, fork, Explore and Plan, and a seventh lane.
2. `task gate T=...`: any tasks as gates, one line each, the decisive lines of a failure, the
   full log in `.cache/gate/` for a targeted Read.
3. `task outline F=...`: a file's functions, headings or tasks with line numbers, so a Read
   takes `offset` and `limit`.
4. `task eval:synthetic:diff`: means with deltas and only the clips that moved.
5. `task claude:tokens` and its `:agents`, `:reads`, `:tools` views: where sessions spent.
6. `.claude/settings.json` holds `Bash(task *)` and no per-task rules; `claude:allow:check`
   fails on a redundant rule, a deny without its `rtk` twin, or an unwired lane hook.

## Consequences

Each new kind of job costs a script and a test before it is done the first time. Work that
is truly one-off, reading an unfamiliar upstream source to decide whether to adopt it, still
needs a Read; the rule is to read a part, not to never read.

The lane count is kept by hooks, not by Claude Code. Six Agent calls in one message reserve
slots at PreToolUse; a crashed session's lanes expire after three hours, or
`task claude:lanes:reset` frees them. The hook events SubagentStart and SubagentStop must
exist in the installed Claude Code; without them, reservations expire after two minutes and
the limit counts only agents started in that window.

The allowlist is no longer an index of tasks. `task --list` is.

## Alternatives considered

1. **Instructions only.** CLAUDE.md already said "ask the local tools first"; the sessions
   above happened anyway. A refusal from a hook is not skipped under pressure.
2. **rtk filters for every noisy task.** `.rtk/filters.toml` is pinned by hash and a person
   re-trusts it after each edit, and a filter strips lines without saving them. `task gate`
   keeps the full log and needs no trust step.
3. **A general-purpose agent with a strict prompt.** The prompt is the caller's to write on
   each dispatch, so it drifts; a named agent's contract is a file under review.
