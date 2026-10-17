#!/usr/bin/env python3
"""Claude Code hooks for this repository.

CLAUDE.md requires that an edit leave the code linted, the tests green and the documentation
honest. An instruction is not a mechanism: this makes it one. Editing source, a Taskfile or a
document marks the tree dirty, and the turn cannot end until `task validate` is clean.

A second mechanism guards the allowance. An agent started at 96% of a usage window dies
mid-write and loses what it was holding, so a new agent is refused once any window is at the
ceiling, with the time the session comes back.

Events, read as JSON on stdin:
  post-tool-use  after Edit/Write/MultiEdit - marks the tree dirty when the file is in scope
  pre-tool-use   before Agent - refuses a new agent while a usage window is at the ceiling
  stop           at the end of a turn - validates if dirty, blocks on failure

The marker lives under .git, which is never committed.

Exit codes follow the hook contract: 0 lets the turn end, and a Stop hook returns its block
decision as JSON on stdout rather than as a non-zero exit.
"""

import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT / ".git" / "beat-analyser-dirty"

CEILING = 95.0
USAGE_CACHE = ROOT / ".git" / "claude-usage.json"
# The usage endpoint answers 429 to a tight poll. One reading serves every check inside this.
CACHE_SECONDS = 60

WATCHED_SUFFIXES = (".yml", ".yaml", ".py", ".md", ".sh", ".html", ".js", ".css", ".toml")
SKIP = ("graphify-out/", "uploads/", ".claude/skills/", ".venv/")

# Reasons the validation could not run at all. None of them is a lint failure, and blocking
# a turn on one walls the agent in behind something no edit can fix.
DID_NOT_RUN = (
    "not on PATH",
    "command not found",
    "precondition not met",
    "executable file not found",
    "No module named pytest",
    # tests/conftest.py imports hypothesis; a .venv from before it was added lacks it.
    "No module named 'hypothesis'",
)


def payload() -> dict:
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}


def in_scope(path: str) -> bool:
    relative = path.replace("\\", "/")
    root = ROOT.as_posix()
    if root.lower() in relative.lower():
        relative = relative[relative.lower().index(root.lower()) + len(root) :].lstrip("/")
    if not relative.endswith(WATCHED_SUFFIXES):
        return False
    return not any(skip in relative for skip in SKIP)


def post_tool_use(event: dict) -> int:
    path = str(event.get("tool_input", {}).get("file_path", ""))
    if not path or not in_scope(path):
        return 0

    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.touch()
    return 0


def run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, shell=False, check=False
    )


def usage_reading() -> dict:
    """The allowance, from a cached reading where one is fresh enough to reuse.

    An unreadable endpoint answers an empty dict: not knowing the allowance never blocks work.
    """
    try:
        cached = json.loads(USAGE_CACHE.read_text(encoding="utf-8"))
        if time.time() - float(cached.get("read_at", 0)) < CACHE_SECONDS:
            return cached.get("reading", {})
    except (OSError, ValueError, TypeError):
        pass

    budget = ROOT / "tasks" / "claude" / "budget.py"
    result = subprocess.run(
        [sys.executable, str(budget), "--json", "--ceiling", str(CEILING)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        reading = json.loads(result.stdout)
    except (ValueError, TypeError):
        return {}
    if not reading.get("ok"):
        return {}

    # A cache that cannot be written costs a reading, not a run.
    with contextlib.suppress(OSError):
        USAGE_CACHE.write_text(
            json.dumps({"read_at": time.time(), "reading": reading}), encoding="utf-8"
        )
    return reading


def span(seconds: int) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60}m" if hours else f"{rest // 60}m"


def pre_tool_use(event: dict) -> int:
    if event.get("tool_name") != "Agent":
        return 0

    reading = usage_reading()
    if not reading.get("over_ceiling"):
        return 0

    spent = ", ".join(
        "{} at {:.0f}%".format(row["label"], row["utilization"])
        for row in reading.get("windows", [])
        if row["utilization"] >= CEILING
    )
    reason = (
        f"Usage is at the {CEILING:g}% ceiling ({spent}). Do not dispatch this agent: one "
        "started now dies mid-write and loses what it was holding.\n"
        f"Write down where the work stands and resume in "
        f"{span(reading.get('wait_seconds', 0))} with `task claude:resume`."
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


def stop(event: dict) -> int:
    # A blocked stop re-enters this hook once the agent replies. Without this the pair loops.
    if event.get("stop_hook_active"):
        return 0
    if not MARKER.exists():
        return 0

    result = run("task", "validate")
    output = result.stdout + result.stderr

    if result.returncode != 0:
        # A missing tool is not a lint failure. The marker stays set, so validation still
        # happens on the first turn that ends on a machine able to run it.
        reason = next((r for r in DID_NOT_RUN if r in output), None)
        if reason is not None:
            print(
                json.dumps(
                    {
                        "systemMessage": "Skipped task validate - it could not run "
                        f"({reason}). Run task dev:setup. The marker stays set, so "
                        "validation happens on the next turn that can run it."
                    }
                )
            )
            return 0
        tail = output.strip().splitlines()[-40:]
        print(
            json.dumps(
                {
                    "decision": "block",
                    "reason": "task validate failed. Fix it before ending the turn:\n"
                    + "\n".join(tail),
                }
            )
        )
        return 0

    MARKER.unlink(missing_ok=True)
    # AST-only, no API cost. A missing graphify binary is not an error here.
    with contextlib.suppress(OSError):
        run("graphify", "update", ".", "--force")
    return 0


def main() -> int:
    event_name = sys.argv[1] if len(sys.argv) > 1 else ""
    event = payload()

    if event_name == "post-tool-use":
        return post_tool_use(event)
    if event_name == "pre-tool-use":
        return pre_tool_use(event)
    if event_name == "stop":
        return stop(event)

    print(f"unknown hook event '{event_name}'", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
