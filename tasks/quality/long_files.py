#!/usr/bin/env python3
"""File-length ratchet over the tracked source, Taskfiles and documents.

Vendored from the toolkit repository, retargeted. A 300-line module is one nobody reads in
full before editing, agents included. This compares against a committed baseline: a file
already over may shrink or hold, never grow, and a file crossing for the first time fails.

Only tracked (or new, unignored) files under app/, tasks/, docs/ and tests/ count.

Usage:
  python3 tasks/quality/long_files.py                    # files at or over the limit
  python3 tasks/quality/long_files.py --top 20           # longest files and their headroom
  python3 tasks/quality/long_files.py --check            # fail on a new or grown offender
  python3 tasks/quality/long_files.py --update-baseline  # accept the current lengths
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "long-files.baseline"
LIMIT = 300
ROOTS = ("app", "tasks", "docs", "tests")
SUFFIXES = (".py", ".yml", ".yaml", ".md", ".sh", ".html", ".js", ".css")
# Implementation plans carry every step's code in full; they are executed, not maintained.
EXCLUDED = ("graphify-out/", "docs/superpowers/")


def tracked_files() -> list[str]:
    # -co --exclude-standard, not a bare ls-files: a document added but not yet
    # committed is precisely the one whose paths have never been checked. The bare
    # form lists only what git already tracks, so a new README was invisible to the
    # gate until after it had been committed.
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", *ROOTS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        line
        for line in out.splitlines()
        if line.endswith(SUFFIXES) and not any(skip in line for skip in EXCLUDED)
    ]


def line_count(path: str) -> int | None:
    """Lines in a file, or None when git lists it and the working tree does not have it.

    That happens for a tracked file deleted but not yet committed - a real state during a
    refactor, and not a reason for the gate to crash.
    """
    try:
        with (ROOT / path).open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def measure() -> dict[str, int]:
    """Only files at or over the limit. Below it there is nothing to ratchet."""
    lengths = {}
    for path in tracked_files():
        count = line_count(path)
        if count is not None and count >= LIMIT:
            lengths[path] = count
    return lengths


def read_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    accepted = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, _, count = line.rpartition(" ")
        if path and count.isdigit():
            accepted[path.strip()] = int(count)
    return accepted


def write_baseline(lengths: dict[str, int]) -> None:
    header = (
        f"# Files at or over the {LIMIT}-line ceiling, accepted at these lengths.\n"
        "# Regenerate with: task quality:long-files:baseline\n"
        "# A file here may shrink. Growth, or a new entrant, fails "
        "task quality:long-files:check.\n"
    )
    body = "".join(f"{path} {count}\n" for path, count in sorted(lengths.items()))
    BASELINE.write_text(header + body, encoding="utf-8")


def report_top(count: int) -> int:
    lengths = {
        path: count
        for path, count in ((p, line_count(p)) for p in tracked_files())
        if count is not None
    }
    ranked = sorted(lengths.items(), key=lambda item: item[1], reverse=True)[:count]
    if not ranked:
        print("no tracked files in scope")
        return 0
    width = max(len(path) for path, _ in ranked)
    for path, length in ranked:
        headroom = LIMIT - length
        note = f"{headroom:+d}" if headroom < 0 else f"{headroom} left"
        print(f"  {path:<{width}}  {length:>4}  {note}")
    return 0


def main() -> int:
    if "--top" in sys.argv:
        index = sys.argv.index("--top") + 1
        wanted = int(sys.argv[index]) if index < len(sys.argv) else 20
        return report_top(wanted)

    lengths = measure()

    if "--update-baseline" in sys.argv:
        write_baseline(lengths)
        print(f"baseline written: {BASELINE.relative_to(ROOT)} ({len(lengths)} file(s))")
        return 0

    baseline = read_baseline()
    # Two ways to fail: a file the baseline has never seen, and one that grew past what the
    # baseline accepted. Both are reported with the number that makes the fix obvious.
    offenders = []
    for path, length in sorted(lengths.items()):
        accepted = baseline.get(path)
        if accepted is None:
            offenders.append(f"{path}: {length} lines, over the {LIMIT}-line ceiling (new)")
        elif length > accepted:
            offenders.append(f"{path}: {length} lines, grown from {accepted}")

    if "--check" not in sys.argv:
        for path, length in sorted(lengths.items(), key=lambda item: item[1], reverse=True):
            accepted = baseline.get(path)
            mark = "NEW  " if accepted is None else "     "
            print(
                f"{mark}{path}: {length} lines"
                + ("" if accepted is None else f" (baseline {accepted})")
            )
        print(f"\n{len(lengths)} file(s) at or over {LIMIT} lines, {len(offenders)} failing")
        return 0

    if offenders:
        print("\n".join(offenders), file=sys.stderr)
        print(
            f"\n{len(offenders)} file(s) over the ceiling or grown past baseline", file=sys.stderr
        )
        return 1

    shrunk = sum(1 for path, length in lengths.items() if length < baseline.get(path, length))
    gone = len(set(baseline) - set(lengths))
    print(f"{len(lengths)} file(s) at or over {LIMIT} lines, none new, none grown.")
    if shrunk or gone:
        print(
            f"{shrunk} shrank and {gone} dropped below the ceiling; "
            "run task quality:long-files:baseline to lower it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
