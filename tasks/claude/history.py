#!/usr/bin/env python3
"""What this project's Claude Code sessions actually ran, and which of it needed approval.

Every session is a .jsonl transcript under ~/.claude/projects/<slug>/. Each Bash or PowerShell
tool call records its command, so the transcripts are a usage log: the commands that repeat are
the ones worth a task, and the ones the allowlist in .claude/settings.json does not cover are
the ones that interrupted the person with an approval prompt.

Vendored from the toolkit repository; `--prompts` is new here.

Usage:
  python3 tasks/claude/history.py                 # top commands, and what has no task yet
  python3 tasks/claude/history.py --top 40        # more rows
  python3 tasks/claude/history.py --inline        # ad-hoc python/jq one-liners, longest first
  python3 tasks/claude/history.py --prompts       # commands the allowlist would not cover
  python3 tasks/claude/history.py --sessions      # per-session transcript sizes
"""

import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / ".claude" / "settings.json"
# Claude Code names a project's transcript folder after its path, every non-alphanumeric
# character a dash.
SLUG = re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
TRANSCRIPTS = Path.home() / ".claude" / "projects" / SLUG

SHAPE = re.compile(r"^\s*([A-Za-z0-9_./\\-]+(?:\s+[A-Za-z0-9:_.-]+){0,2})")
INLINE = re.compile(r"\b(python|python3|python\.exe|jq)\b\s+(-W \S+\s+)?(-c|-r)\b")
# A chain or pipe is checked part by part by the permission system, and any multi-line or
# compound command is far more likely to prompt. Counted separately so the habit is visible.
COMPOUND = re.compile(r"&&|\|\||;|\||\n|\$\(")


def commands() -> list[tuple[str, str]]:
    """(tool, command) for every Bash and PowerShell call in every transcript."""
    found = []
    if not TRANSCRIPTS.is_dir():
        return found
    for transcript in sorted(TRANSCRIPTS.glob("*.jsonl")):
        for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"Bash"' not in line and '"PowerShell"' not in line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            for block in _content(entry):
                tool = block.get("name")
                if block.get("type") != "tool_use" or tool not in ("Bash", "PowerShell"):
                    continue
                command = (block.get("input") or {}).get("command")
                if isinstance(command, str) and command.strip():
                    found.append((tool, command))
    return found


def _content(entry: dict) -> list[dict]:
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def shape_of(command: str) -> str:
    first = command.strip().splitlines()[0]
    match = SHAPE.match(first)
    return match.group(1).strip() if match else first[:60]


def allow_patterns() -> dict[str, list[str]]:
    """`Bash(task *)` style rules from settings.json, as fnmatch patterns per tool."""
    try:
        allow = json.loads(SETTINGS.read_text(encoding="utf-8"))["permissions"]["allow"]
    except (OSError, ValueError, KeyError):
        return {}
    patterns: dict[str, list[str]] = {}
    for rule in allow:
        match = re.fullmatch(r"(Bash|PowerShell)\((.*)\)", rule)
        if match:
            patterns.setdefault(match.group(1), []).append(match.group(2))
    return patterns


def needs_approval(tool: str, command: str, patterns: dict[str, list[str]]) -> bool:
    if COMPOUND.search(command.strip()):
        return True
    text = command.strip()
    return not any(fnmatch.fnmatchcase(text, p) for p in patterns.get(tool, []))


def known_tasks() -> set[str]:
    names = set()
    for taskfile in [ROOT / "Taskfile.yml", *ROOT.glob("tasks/*/Taskfile.yml")]:
        domain = taskfile.parent.name if taskfile.parent != ROOT else ""
        for line in taskfile.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^  ([a-z][a-z0-9:_-]*):\s*$", line)
            if match:
                names.add(f"{domain}:{match.group(1)}" if domain else match.group(1))
    return names


def report_prompts(found: list[tuple[str, str]]) -> int:
    patterns = allow_patterns()
    flagged = [(tool, cmd) for tool, cmd in found if needs_approval(tool, cmd, patterns)]
    compound = sum(1 for _, cmd in flagged if COMPOUND.search(cmd.strip()))
    print(
        f"{len(flagged)} of {len(found)} command(s) not covered by the allowlist "
        f"({compound} of them chained, piped or multi-line)\n"
    )
    counts = Counter(f"{tool}: {shape_of(cmd)}" for tool, cmd in flagged)
    for shape, count in counts.most_common(30):
        print(f"{count:5d}  {shape}")
    print("\nFix: run one allowlisted command per call, or put the operation behind a task.")
    return 0


def main() -> int:
    found = commands()
    if not found:
        print(f"no transcripts under {TRANSCRIPTS}", file=sys.stderr)
        return 1

    if "--sessions" in sys.argv:
        for transcript in sorted(TRANSCRIPTS.glob("*.jsonl")):
            print(f"{transcript.stat().st_size / 1_000_000:8.1f} MB  {transcript.name}")
        return 0

    if "--prompts" in sys.argv:
        return report_prompts(found)

    if "--inline" in sys.argv:
        inline = sorted({c for _, c in found if INLINE.search(c)}, key=len, reverse=True)
        print(f"{len(inline)} ad-hoc one-liner(s) - each one is a script not yet written\n")
        for command in inline[:20]:
            print(f"  {' '.join(command.split())[:150]}")
        return 0

    position = sys.argv.index("--top") + 1 if "--top" in sys.argv else -1
    top = int(sys.argv[position]) if 0 < position < len(sys.argv) else 25

    counts = Counter(shape_of(c) for _, c in found)
    tasks = known_tasks()
    sessions = len(list(TRANSCRIPTS.glob("*.jsonl")))
    print(f"{len(found)} shell command(s) across {sessions} session(s)\n")
    for shape, count in counts.most_common(top):
        print(f"{count:5d}  {shape}")

    print("\nrepeated and not behind a task:")
    shown = 0
    for shape, count in counts.most_common():
        if count < 3 or shape.startswith("task ") or shape.split()[0] in ("cd", "echo", "ls"):
            continue
        if any(part in tasks for part in shape.split()[1:2]):
            continue
        print(f"{count:5d}  {shape}")
        shown += 1
        if shown >= 15:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
