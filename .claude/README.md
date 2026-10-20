# .claude

Project Claude Code configuration, modelled on the toolkit repository. Every listed skill,
agent and command costs listing tokens in every session: add one only when
[CLAUDE.md](../CLAUDE.md) calls for it.

## Files

| Path | What it is |
| --- | --- |
| `settings.json` | Models, permissions, hooks, plugins. Tracked. |
| `settings.local.json` | Per-machine overrides: skill toggles, status line, extra read paths. Gitignored. |
| `rules/taskfile-shell.md` | Loaded when a Taskfile is touched: go-task's shell is not bash |
| `agents/*.md` | Named single-job subagents; roster in [CLAUDE.md](../CLAUDE.md). Each calls tasks and returns a receipt of at most 10 lines ([ADR 0011](../docs/adr/0011-tool-first-agent-loop.md)) |
| `skills/humanizer/` | Strip AI-writing tells from prose. Vendored from [blader/humanizer](https://github.com/blader/humanizer) |

## Plugins

Enabled in `settings.json`, fetched from their marketplace, not vendored.
`task claude:skills:setup` registers caveman and superpowers at user scope as well.

| Plugin | Source | Gives |
| --- | --- | --- |
| `superpowers` | `claude-plugins-official` ([obra/superpowers](https://github.com/obra/superpowers)) | Brainstorming, TDD, systematic debugging, planning, verification skills |
| `caveman` | `caveman` ([JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman)) | Compressed reply prose, `/caveman-*` skills, `cavecrew-*` agents |
| `code-review` | `claude-plugins-official` | `/code-review` |
| `feature-dev` | `claude-plugins-official` | `/feature-dev` and its explorer, architect, reviewer agents |
| `security-guidance` | `claude-plugins-official` | Pattern warnings on edit, diff review on Stop |
| `claude-code-setup` | `claude-plugins-official` | Configuration helpers |

## Hooks

All in [tasks/claude/hooks.py](../tasks/claude/hooks.py), invoked through
`$CLAUDE_PROJECT_DIR` so they work whatever the shell's current directory:

1. `PostToolUse` on Edit/Write marks the tree dirty when source, a Taskfile or a doc changes.
2. `Stop` runs `task validate` while dirty and blocks the turn on failure.
3. `PreToolUse` on Agent refuses a new agent at 95 percent of a usage window, a catch-all
   agent type (general-purpose, fork, Explore, Plan), or a seventh concurrent lane.
4. `SubagentStart` and `SubagentStop` keep the lane count (`tasks/claude/lanes.py`).

## Permissions

`allow` covers every task through `Bash(task *)`, plus `git`, `rtk`, the venv Python, the
linters, read-only Docker commands, `curl` to localhost and read-only shell tools. No
per-task and no PowerShell rules: `task claude:allow:check` fails on either. `deny` wins
over `allow`: `sudo`, `git push`, `git reset --hard`, `git clean`, `docker push`, image and
volume removal, `task uploads:clean`, `task git:commit`, `task claude:allow:sync` and
deleting `uploads/`, each task deny twinned with its `rtk task` form. `python *` executes arbitrary code, so the deny
list stops an agent that follows the rules, not one that routes around them.

## Rules

1. Vendored files stay verbatim; refresh from upstream rather than editing.
2. A vendored skill is un-ignored by name in [.gitignore](../.gitignore).
3. Globally installed skills (graphify) are not copied here.
