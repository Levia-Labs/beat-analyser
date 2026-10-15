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
3. `PreToolUse` on Agent refuses a new agent at 95 percent of a usage window.

## Permissions

`allow` covers `task`, `git`, `rtk`, the venv Python, the linters, read-only Docker
commands, `curl` to localhost and read-only shell tools. `deny` wins over `allow`: `sudo`,
`git push`, `git reset --hard`, `git clean`, `docker push`, image and volume removal,
`task uploads:clean` and deleting `uploads/`. `python *` executes arbitrary code, so the deny
list stops an agent that follows the rules, not one that routes around them.

## Rules

1. Vendored files stay verbatim; refresh from upstream rather than editing.
2. A vendored skill is un-ignored by name in [.gitignore](../.gitignore).
3. Globally installed skills (graphify) are not copied here.
