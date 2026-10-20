#!/usr/bin/env python3
"""Run tasks as gates: one line per gate, and only the decisive lines of a failure.

A model deciding what to do next needs pass or fail, and on a failure the lines that say why.
`task rust:test` prints the whole cargo build, `task docker:e2e` the whole image build:
hundreds of lines paid for on every later turn (ADR 0011). This keeps the full log in
.cache/gate/<task>.log, for a Read with offset when the excerpt is not enough.

Usage:
  python3 tasks/dev/gate.py "rust:test" "test K=notes" "types:check"
  python3 tasks/dev/gate.py --lines 60 "docker:e2e"

Exit 1 when any gate failed. Every gate runs: one failure does not hide the next.
"""

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / ".cache" / "gate"

# Lines that carry a failure's reason in the tools this repository runs: pytest, ruff,
# basedpyright, cargo, clippy, ast-grep, go-task, docker, Python tracebacks.
DECISIVE = re.compile(
    r"error|fail|panicked|assert|traceback|exception|warning:|^\s*--> |^E\s|"
    r"^\S+:\d+:\d*:? |^\s+File \"|not found|denied|refused|missing|timed out",
    re.IGNORECASE,
)
# Build progress, and pytest progress rows without an F or E in them.
NOISE = re.compile(
    r"^\s*(Compiling|Downloaded|Downloading|Fresh|Checking) |^#\d+ |^[.s]+\s+\[\s*\d+%\]$"
)


def excerpt(output: str, limit: int) -> list[str]:
    """The decisive lines of a failed run and its last ten, in their original order."""
    lines = [
        line.rstrip() for line in output.splitlines() if line.strip() and not NOISE.search(line)
    ]
    keep = {n for n, line in enumerate(lines) if DECISIVE.search(line)}
    if not keep:
        return lines[-limit:]
    keep |= set(range(max(0, len(lines) - 10), len(lines)))
    return [lines[n] for n in sorted(keep)][:limit]


def log_name(gate: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", gate).strip("_") + ".log"


def run(gate: str, limit: int) -> bool:
    started = time.monotonic()
    result = subprocess.run(
        ["task", *shlex.split(gate)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    seconds = time.monotonic() - started
    output = result.stdout + result.stderr
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / log_name(gate)
    log.write_text(output, encoding="utf-8")
    if result.returncode == 0:
        print(f"ok    {gate}  {seconds:.0f}s")
        return True
    where = log.relative_to(ROOT).as_posix()
    print(f"FAIL  {gate}  {seconds:.0f}s  exit {result.returncode}  log {where}")
    for line in excerpt(output, limit):
        print(f"  {line[:200]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("gates", nargs="+", help="a task name with its vars, one per argument")
    parser.add_argument("--lines", type=int, default=30, help="excerpt lines per failure")
    args = parser.parse_args()
    results = [run(gate, args.lines) for gate in args.gates]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
