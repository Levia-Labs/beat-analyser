#!/usr/bin/env python3
"""Keep an explicit allow rule for every task in .claude/settings.json.

`Bash(task *)` already covers every task; the explicit rules make the allowlist readable as
an index of what an agent may run, and keep working if the wildcard is ever narrowed. A
task with a deny rule (`git:commit`, `uploads:clean`, ...) gets no allow rule: deny wins
anyway, and listing it as allowed would mislead a reader.

Rewriting the permission file is an approval prompt in Claude Code by design - an agent must
not widen its own permissions silently - so agents are denied `claude:allow:sync` and a
person runs it. `task validate` runs the check with --warn: `Bash(task *)` already allows a
new task, so a missing explicit rule is reported, never a failure.

Usage:
  python3 tasks/claude/allow_tasks.py                  # rewrite the rules (a person runs this)
  python3 tasks/claude/allow_tasks.py --check          # exit 1 when a task has no rule
  python3 tasks/claude/allow_tasks.py --check --warn   # report only, exit 0
"""

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / ".claude" / "settings.json"
# Bash only: agents in this repository do not use PowerShell (it is denied outright).
TOOLS = ("Bash",)
WILDCARDS = {f"{tool}(task *)" for tool in TOOLS}
PER_TASK = re.compile(r"^(Bash|PowerShell)\(task [a-z0-9:_-]+( \*)?\)$")


def task_names() -> list[str]:
    out = subprocess.run(
        ["task", "--list-all", "--json"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return sorted(t["name"] for t in json.loads(out)["tasks"])


def denied(name: str, deny: list[str]) -> bool:
    for rule in deny:
        match = re.fullmatch(r"(?:Bash|PowerShell)\((.*)\)", rule)
        if match and (
            fnmatch.fnmatchcase(f"task {name}", match.group(1))
            or fnmatch.fnmatchcase(f"task {name} X", match.group(1))
        ):
            return True
    return False


def wanted(names: list[str], deny: list[str]) -> list[str]:
    rules = []
    for tool in TOOLS:
        for name in names:
            if not denied(name, deny):
                rules += [f"{tool}(task {name})", f"{tool}(task {name} *)"]
    return rules


def main() -> int:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    permissions = settings["permissions"]
    rules = wanted(task_names(), permissions.get("deny", []))

    if "--check" in sys.argv:
        missing = [r for r in rules if r not in permissions["allow"]]
        for rule in missing:
            print(f"  missing allow rule: {rule}")
        if missing:
            print(f"{len(missing)} rule(s) missing. A person runs: task claude:allow:sync")
            return 0 if "--warn" in sys.argv else 1
        print(f"every task has an allow rule ({len(rules)} rules)")
        return 0

    others = [r for r in permissions["allow"] if not PER_TASK.match(r) and r not in WILDCARDS]
    permissions["allow"] = sorted(WILDCARDS) + rules + others
    SETTINGS.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rules)} per-task allow rule(s) written to {SETTINGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
