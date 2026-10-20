# tasks / claude

Namespace `claude:`. Nothing else in the repository needs the Claude Code CLI.

| Script | Role |
| --- | --- |
| `hooks.py` | Wired in [.claude/settings.json](../../.claude/settings.json). `post-tool-use` marks the tree dirty when source, a Taskfile or a document is edited; `stop` runs `task validate` while dirty and blocks the turn on failure, then refreshes the graph; `pre-tool-use` refuses a new agent while a usage window is at 95 percent, then asks `lanes.py`; `subagent-start` and `subagent-stop` keep the lane count |
| `history.py` | Reads this project's session transcripts. `claude:history` ranks commands and lists repeats with no task; `claude:history:prompts` lists commands the allowlist does not cover (each one was an approval prompt); `claude:history:inline` lists ad-hoc one-liners. Vendored from the toolkit repository, `--prompts` added |
| `lanes.py` | Called by `hooks.py`: refuses catch-all agent types and a seventh concurrent lane, counting through PreToolUse, SubagentStart and SubagentStop ([ADR 0011](../../docs/adr/0011-tool-first-agent-loop.md)). `claude:lanes` lists lanes, `claude:lanes:reset` forgets them |
| `tokens.py` | Reads the same transcripts for cost: `claude:tokens` per session and subagent, `:agents`, `:reads` (files Read most, characters fed back), `:tools` (result size by tool). `PROJECT=` points it at another repository |
| `allow_tasks.py` | Keeps `.claude/settings.json` minimal: `Bash(task *)` and no per-task or PowerShell rule, every task deny twinned with `rtk task`, lane hooks wired. `claude:allow:sync` rewrites it; a person runs it, since rewriting the permission file always prompts and agents are denied it. `claude:allow:check` fails on a finding; `claude:allow:report`, run by `task validate`, lists them silently when clean |
| `budget.py` | Usage windows against a ceiling, read from the endpoint behind the CLI's `/usage`. Vendored from the toolkit repository |
| `skills.yml` | `claude:skills:setup` adds the caveman and superpowers marketplaces and installs both; `claude:skills:verify` reports what is registered |

Hooks never wall an agent in: an unreadable usage endpoint does not block, and a validation
that could not run (missing tool) is skipped with a message while the dirty marker stays.
