#!/usr/bin/env python3
"""Subagent lanes: which agents may run, and how many at once (ADR 0011).

A catch-all agent (general-purpose, fork, Explore, Plan) is handed a goal and reads its way
to an answer: every file it opens is context it pays for again on each turn, and it pastes
what it found back into the parent. Sessions here spent hundreds of thousands of tokens that
way. A named agent from .claude/agents/ has one job, calls tasks, and returns a receipt.
So a catch-all is refused, and so is a seventh concurrent lane.

hooks.py calls this on three events:
  PreToolUse(Agent)  refusal() - type and concurrency; a pass reserves a slot
  SubagentStart      started() - the reservation becomes a running lane
  SubagentStop       stopped() - the lane is freed

A reservation covers the gap between the PreToolUse and the SubagentStart, so six Agent
calls in one message cannot all pass on an empty count. Entries expire, so a crashed
session never holds a slot for good. State lives under .git, never committed.

Usage:
  python3 tasks/claude/lanes.py           # running lanes and pending reservations
  python3 tasks/claude/lanes.py --reset   # forget every lane (after a crash)
"""

import contextlib
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".git" / "claude-lanes.json"
LOCK = STATE.with_suffix(".lock")

LIMIT = 6
# An Agent call without subagent_type runs general-purpose.
CATCH_ALL = frozenset({"", "general-purpose", "fork", "claude", "Explore", "Plan"})
RESERVED_SECONDS = 120
STALE_SECONDS = 3 * 3600


def empty() -> dict:
    return {"running": {}, "pending": {}}


def prune(state: dict, now: float) -> dict:
    return {
        "running": {k: t for k, t in state.get("running", {}).items() if now - t < STALE_SECONDS},
        "pending": {
            k: t for k, t in state.get("pending", {}).items() if now - t < RESERVED_SECONDS
        },
    }


def in_use(state: dict) -> int:
    return len(state["running"]) + len(state["pending"])


def refusal(tool_input: dict, state: dict) -> str | None:
    """Why this Agent call may not run, or None when it may."""
    kind = str(tool_input.get("subagent_type") or "")
    if kind in CATCH_ALL:
        return (
            f"Agent type '{kind or 'general-purpose'}' is a catch-all and is refused (ADR 0011). "
            "Use a named single-job agent from .claude/agents/ (listed in CLAUDE.md), or "
            "write one first: a job, the tasks it calls, a receipt of at most 10 lines."
        )
    if in_use(state) >= LIMIT:
        return (
            f"{in_use(state)} of {LIMIT} subagent lanes are in use. Wait for one to finish. "
            "`task claude:lanes` lists them; `task claude:lanes:reset` clears lanes a crashed "
            "session left behind."
        )
    return None


def reserve(state: dict, key: str, now: float) -> None:
    state["pending"][key] = now


def started(state: dict, agent_id: str, now: float) -> None:
    if state["pending"]:
        oldest = min(state["pending"], key=state["pending"].__getitem__)
        del state["pending"][oldest]
    state["running"][agent_id] = now


def stopped(state: dict, agent_id: str) -> None:
    state["running"].pop(agent_id, None)


@contextlib.contextmanager
def locked_state() -> Iterator[dict]:
    """The state, read and written under an exclusive lock file: parallel hooks race."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        try:
            handle = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            # A lock older than a few seconds belongs to a killed hook.
            with contextlib.suppress(OSError):
                if time.time() - LOCK.stat().st_mtime > 10:
                    LOCK.unlink()
            time.sleep(0.05)
    else:
        handle = None
    try:
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = empty()
        state = prune(state, time.time())
        yield state
        with contextlib.suppress(OSError):
            STATE.write_text(json.dumps(state), encoding="utf-8")
    finally:
        if handle is not None:
            os.close(handle)
            with contextlib.suppress(OSError):
                LOCK.unlink()


def main() -> int:
    with locked_state() as state:
        if "--reset" in sys.argv:
            state.clear()
            state.update(empty())
            print("lanes reset")
            return 0
        print(f"{in_use(state)} of {LIMIT} lanes in use")
        now = time.time()
        for agent, since in state["running"].items():
            print(f"  running  {agent}  {int(now - since) // 60}m")
        for key, since in state["pending"].items():
            print(f"  pending  {key}  {int(now - since)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
