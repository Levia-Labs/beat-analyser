# tasks / claude

Namespace `claude:`. Nothing else in the repository needs the Claude Code CLI.

| Script | Role |
| --- | --- |
| `hooks.py` | Wired in [.claude/settings.json](../../.claude/settings.json). `post-tool-use` marks the tree dirty when source, a Taskfile or a document is edited; `stop` runs `task validate` while dirty and blocks the turn on failure, then refreshes the graph; `pre-tool-use` refuses a new agent while a usage window is at 95 percent |
| `history.py` | Reads this project's session transcripts. `claude:history` ranks commands and lists repeats with no task; `claude:history:prompts` lists commands the allowlist does not cover (each one was an approval prompt); `claude:history:inline` lists ad-hoc one-liners. Vendored from the toolkit repository, `--prompts` added |
| `allow_tasks.py` | `claude:allow:sync` writes an allow rule (`task NAME` and `task NAME *`, Bash only - the PowerShell tool is denied) for every task into `.claude/settings.json`, skipping denied ones. A person runs it: rewriting the permission file always prompts in Claude Code, so agents are denied it. `claude:allow:check` fails on a missing rule; `claude:allow:report`, run by `task validate`, only lists them (`Bash(task *)` already allows every task) |
| `budget.py` | Usage windows against a ceiling, read from the endpoint behind the CLI's `/usage`. Vendored from the toolkit repository |
| `skills.yml` | `claude:skills:setup` adds the caveman and superpowers marketplaces and installs both; `claude:skills:verify` reports what is registered |

Hooks never wall an agent in: an unreadable usage endpoint does not block, and a validation
that could not run (missing tool) is skipped with a message while the dirty marker stays.
