# .claude

Project Claude Code configuration, modelled on the toolkit repository. Every listed skill,
agent and command costs listing tokens in every session: add one only when
[CLAUDE.md](../CLAUDE.md) calls for it.

## Files

| Path                             | What it is                                                                                               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `settings.json`                  | Models, permissions, hooks, plugins. Tracked.                                                            |
| settings.local.json (gitignored) | Per-machine overrides: skill toggles, status line, extra read paths. Gitignored.                         |
| `agents/beatcore-tuner.md`       | The pillar-only subagent; the shared agents come from the crepidoma plugin                               |
| `skills/humanizer/`              | Strip AI-writing tells from prose. Vendored from [blader/humanizer](https://github.com/blader/humanizer) |

## Plugins

Enabled in `settings.json`, fetched from their marketplace, not vendored.
`task claude:skills:setup` registers caveman and superpowers at user scope as well.

| Plugin            | Source                                                                              | Gives                                                                         |
| ----------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| crepidoma         | `levia-labs` ([Levia-Labs/Crepidoma](https://github.com/Levia-Labs/Crepidoma))      | Shared directives, agents, hooks; opts the repo in to the plugin's task gates |
| superpowers       | `claude-plugins-official` ([obra/superpowers](https://github.com/obra/superpowers)) | Brainstorming, TDD, systematic debugging, planning, verification skills       |
| caveman           | `caveman` ([JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman))       | Compressed reply prose, `/caveman-*` skills, `cavecrew-*` agents              |
| code-review       | `claude-plugins-official`                                                           | `/code-review`                                                                |
| feature-dev       | `claude-plugins-official`                                                           | `/feature-dev` and its explorer, architect, reviewer agents                   |
| security-guidance | `claude-plugins-official`                                                           | Pattern warnings on edit, diff review on Stop                                 |
| claude-code-setup | `claude-plugins-official`                                                           | Configuration helpers                                                         |
| frontend-design   | `anthropic-plugin-directory`                                                        | Visual design guidance for UI work                                            |

## Hooks

All come from the crepidoma plugin: the Stop gate runs the repo's `task validate` while the tree
is dirty, agent launches are budget- and lane-checked, raw reads are capped. The committed
`crepidoma@levia-labs` entry in `settings.json` is the opt-in. Each collaborator runs once:
`claude plugin install crepidoma@levia-labs --scope project`.

## Permissions

`allow` covers every task through `Bash(task *)`, plus `git`, `rtk`, the venv Python, the
linters, read-only Docker commands, `curl` to localhost and read-only shell tools. No
per-task and no PowerShell rules: `task claude:allow:check` fails on either. `deny` wins
over `allow`: `sudo`, `git push`, `git reset --hard`, `git clean`, `docker push`, image and
volume removal, `task uploads:clean`, `task git:commit`, and
deleting `uploads/`, each task deny twinned with its `rtk task` form. `python *` executes arbitrary code, so the deny
list stops an agent that follows the rules, not one that routes around them.

## Rules

1. Vendored files stay verbatim; refresh from upstream rather than editing.
2. A vendored skill is un-ignored by name in [.gitignore](../.gitignore).
3. Globally installed skills (graphify) are not copied here.
