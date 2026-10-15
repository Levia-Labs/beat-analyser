# tasks / claude

Namespace `claude:`. Nothing else in the repository needs the Claude Code CLI.

| Script | Role |
| --- | --- |
| `hooks.py` | Wired in [.claude/settings.json](../../.claude/settings.json). `post-tool-use` marks the tree dirty when source, a Taskfile or a document is edited; `stop` runs `task validate` while dirty and blocks the turn on failure, then refreshes the graph; `pre-tool-use` refuses a new agent while a usage window is at 95 percent |
| `budget.py` | Usage windows against a ceiling, read from the endpoint behind the CLI's `/usage`. Vendored from the toolkit repository |
| `skills.yml` | `claude:skills:setup` adds the caveman and superpowers marketplaces and installs both; `claude:skills:verify` reports what is registered |

Hooks never wall an agent in: an unreadable usage endpoint does not block, and a validation
that could not run (missing tool) is skipped with a message while the dirty marker stays.
