#!/usr/bin/env python3
"""Keep .claude/settings.json minimal and its guards wired.

`Bash(task *)` allows every task, and `task --list` is the index of them. The per-task allow
rules this script used to write doubled the file to 480 lines, and sessions read it seven
times (74k characters) to edit it: an index nobody needed, paid for in context (ADR 0011).
The PowerShell tool is denied outright, so a `PowerShell(...)` rule is dead weight too.

What the file must hold:
  1. `Bash(task *)`, and no per-task or PowerShell rule beside it.
  2. Every `Bash(task X)` deny rule twinned as `Bash(rtk task X)`: RTK's hook rewrites a
     command before the permission check sees it.
  3. The lane hooks (SubagentStart, SubagentStop) that tasks/claude/lanes.py counts with.

Rewriting the permission file is an approval prompt in Claude Code by design - an agent must
not widen its own permissions silently - so agents are denied `claude:allow:sync` and a
person runs it. `task validate` runs the check with --warn.

Usage:
  python3 tasks/claude/allow_tasks.py                  # rewrite (a person runs this)
  python3 tasks/claude/allow_tasks.py --check          # exit 1 on a finding
  python3 tasks/claude/allow_tasks.py --check --warn   # report only, exit 0
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / ".claude" / "settings.json"
WILDCARD = "Bash(task *)"
PER_TASK = re.compile(r"^Bash\(task [a-z0-9:_-]+( \*)?\)$")
DEAD = re.compile(r"^PowerShell\(.*\)$")
TASK_DENY = re.compile(r"^Bash\((task .*)\)$")
HOOK = 'python3 "$CLAUDE_PROJECT_DIR/tasks/claude/hooks.py" {}'
LANE_HOOKS = {"SubagentStart": "subagent-start", "SubagentStop": "subagent-stop"}


def twins(deny: list[str]) -> list[str]:
    wanted = []
    for rule in deny:
        match = TASK_DENY.match(rule)
        if match and f"Bash(rtk {match.group(1)})" not in deny:
            wanted.append(f"Bash(rtk {match.group(1)})")
    return wanted


def wired(hooks: dict, event: str) -> bool:
    command = HOOK.format(LANE_HOOKS[event])
    return any(
        h.get("command") == command for group in hooks.get(event, []) for h in group["hooks"]
    )


def findings(settings: dict) -> list[str]:
    allow = settings["permissions"]["allow"]
    deny = settings["permissions"].get("deny", [])
    found = [] if WILDCARD in allow else [f"allow lacks {WILDCARD}"]
    redundant = [r for r in allow if PER_TASK.match(r) and r != WILDCARD]
    dead = [r for r in allow + deny if DEAD.match(r)]
    if redundant:
        found.append(f"{len(redundant)} per-task allow rule(s), covered by {WILDCARD}")
    if dead:
        found.append(f"{len(dead)} PowerShell(...) rule(s); the tool is denied outright")
    found += [f"deny lacks its rtk twin {rule}" for rule in twins(deny)]
    hooks = settings.get("hooks", {})
    found += [f"hook {event} not wired" for event in LANE_HOOKS if not wired(hooks, event)]
    return found


def rewrite(settings: dict) -> dict:
    permissions = settings["permissions"]
    allow = [r for r in permissions["allow"] if not PER_TASK.match(r) and not DEAD.match(r)]
    permissions["allow"] = [WILDCARD] + [r for r in allow if r != WILDCARD]
    deny = [r for r in permissions.get("deny", []) if not DEAD.match(r)]
    permissions["deny"] = deny + twins(deny)
    hooks = settings.setdefault("hooks", {})
    for event, name in LANE_HOOKS.items():
        if not wired(hooks, event):
            hooks.setdefault(event, []).append(
                {"hooks": [{"type": "command", "command": HOOK.format(name)}]}
            )
    return settings


def main() -> int:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))

    if "--check" in sys.argv:
        found = findings(settings)
        for finding in found:
            print(f"  {finding}")
        if found:
            print(f"{len(found)} settings finding(s). A person runs: task claude:allow:sync")
            return 0 if "--warn" in sys.argv else 1
        # Silent when clean under --warn: task validate's output stays its failures.
        if "--warn" not in sys.argv:
            print("settings.json minimal, guards wired")
        return 0

    SETTINGS.write_text(json.dumps(rewrite(settings), indent=2) + "\n", encoding="utf-8")
    print(f"{SETTINGS.relative_to(ROOT)} rewritten: {len(findings(settings))} finding(s) left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
