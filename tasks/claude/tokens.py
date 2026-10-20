#!/usr/bin/env python3
"""Where a project's Claude Code sessions spent their tokens.

history.py counts shell commands. This counts what the model was fed: every tool result is
context that every later turn pays for again. A subagent that Reads twenty files and pastes
them back costs the agent's own turns plus the paste in the parent. The fix for a large row
here is a task whose output is the decision, not the content.

Transcripts: ~/.claude/projects/<slug>/<session>.jsonl, subagents under
<session>/subagents/agent-*.jsonl with a .meta.json naming the agent type.

Usage:
  python3 tasks/claude/tokens.py                    # per session: API tokens, tool result size
  python3 tasks/claude/tokens.py --tools            # tool result chars by tool, all sessions
  python3 tasks/claude/tokens.py --agents           # each subagent: type, tokens, result size
  python3 tasks/claude/tokens.py --reads            # files Read most, with chars fed back
  python3 tasks/claude/tokens.py --session ID       # restrict to one session (prefix)
  python3 tasks/claude/tokens.py --project PATH     # another project's transcripts
  python3 tasks/claude/tokens.py --top N            # rows per table (default 15)
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def transcripts_dir(project: Path) -> Path:
    # Claude Code names the folder after the path, every non-alphanumeric character a dash.
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(project))


@dataclass
class Tally:
    name: str
    kind: str = "session"
    calls: int = 0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0
    tool_calls: Counter = field(default_factory=Counter)
    tool_chars: Counter = field(default_factory=Counter)
    reads: Counter = field(default_factory=Counter)
    read_chars: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.input + self.cache_write + self.cache_read + self.output


def relative(path: str) -> str:
    """One spelling per file: the same file is read as X:\\... and x:/... across sessions."""
    posix = path.replace("\\", "/")
    root = ROOT.as_posix().lower() + "/"
    return posix[len(root) :] if posix.lower().startswith(root) else posix


def _blocks(entry: dict) -> list[dict]:
    message = entry.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _chars(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


def tally(path: Path, name: str, kind: str) -> Tally:
    result = Tally(name=name, kind=kind)
    seen: set[str] = set()
    tool_of: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        # One API response is split across entries, one per content block, each carrying
        # the same usage. Count it once.
        if isinstance(usage, dict) and str(message.get("id")) not in seen:
            seen.add(str(message.get("id")))
            result.calls += 1
            result.input += usage.get("input_tokens", 0) or 0
            result.cache_write += usage.get("cache_creation_input_tokens", 0) or 0
            result.cache_read += usage.get("cache_read_input_tokens", 0) or 0
            result.output += usage.get("output_tokens", 0) or 0
        for block in _blocks(entry):
            if block.get("type") == "tool_use":
                tool = str(block.get("name"))
                target = str((block.get("input") or {}).get("file_path", ""))
                tool_of[str(block.get("id"))] = (tool, target)
                result.tool_calls[tool] += 1
                if tool == "Read" and target:
                    target = relative(target)
                    tool_of[str(block.get("id"))] = (tool, target)
                    result.reads[target] += 1
            elif block.get("type") == "tool_result":
                tool, target = tool_of.get(str(block.get("tool_use_id")), ("?", ""))
                size = _chars(block.get("content"))
                result.tool_chars[tool] += size
                if tool == "Read" and target:
                    result.read_chars[target] += size
    return result


def collect(folder: Path, session: str) -> list[Tally]:
    found = []
    for transcript in sorted(folder.glob("*.jsonl")):
        if session and not transcript.stem.startswith(session):
            continue
        found.append(tally(transcript, transcript.stem[:8], "session"))
        for agent in sorted((folder / transcript.stem / "subagents").glob("agent-*.jsonl")):
            meta = agent.with_suffix(".meta.json")
            label = agent.stem
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                label = f"{info.get('agentType', '?')}: {info.get('description', '')}"
            except (OSError, ValueError):
                pass
            found.append(tally(agent, f"{transcript.stem[:8]} {label}"[:70], "agent"))
    return found


def k(n: int) -> str:
    return f"{n / 1000:,.0f}k"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--session", default="")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--agents", action="store_true")
    parser.add_argument("--reads", action="store_true")
    args = parser.parse_args()

    folder = transcripts_dir(args.project.resolve())
    rows = collect(folder, args.session)
    if not rows:
        print(f"no transcripts under {folder}", file=sys.stderr)
        return 1

    if args.tools:
        calls: Counter = Counter()
        chars: Counter = Counter()
        for row in rows:
            calls.update(row.tool_calls)
            chars.update(row.tool_chars)
        print(f"{'result chars':>12}  {'calls':>5}  tool")
        for tool, size in chars.most_common(args.top):
            print(f"{size:12,d}  {calls[tool]:5d}  {tool}")
        return 0

    if args.reads:
        count: Counter = Counter()
        chars = Counter()
        for row in rows:
            count.update(row.reads)
            chars.update(row.read_chars)
        print(f"{'chars':>9}  {'reads':>5}  file")
        for target, size in chars.most_common(args.top):
            print(f"{size:9,d}  {count[target]:5d}  {target}")
        return 0

    shown = [r for r in rows if r.kind == "agent"] if args.agents else rows
    shown.sort(key=lambda r: r.total, reverse=True)
    by_kind: dict[str, int] = defaultdict(int)
    for row in rows:
        by_kind[row.kind] += row.total
    print(
        f"{len(rows)} transcript(s); sessions {k(by_kind['session'])}, "
        f"subagents {k(by_kind['agent'])} tokens (input+cache+output)"
    )
    print(f"{'total':>7} {'c.read':>7} {'c.write':>7} {'out':>6} {'calls':>5} {'results':>9}  name")
    for row in shown[: args.top]:
        print(
            f"{k(row.total):>7} {k(row.cache_read):>7} {k(row.cache_write):>7} "
            f"{k(row.output):>6} {row.calls:5d} {sum(row.tool_chars.values()):9,d}  {row.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
