#!/usr/bin/env python3
"""Coverage ratchet over app/, from the JSON report task coverage:run writes.

Per file and in total, coverage may hold or rise, never fall below the committed baseline. A
file the baseline has never seen must reach NEW_FILE_FLOOR. Tolerance absorbs float noise
only; the baseline keeps two decimals so a real drop of a hundredth still fails.

Some tests skip without an optional tool (ffmpeg, onnxruntime, a host beatcore), so the
coverage of the files they exercise depends on the machine. The baseline records which tools
were present; on a machine missing one, the files that depend on it are reported as not
compared instead of failing. Run with the venv interpreter so the imports are visible.

Usage:
  python tasks/coverage/coverage.py --check            # fail on a drop
  python tasks/coverage/coverage.py --update-baseline  # accept the current numbers
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / ".cache" / "coverage.json"
BASELINE = Path(__file__).resolve().parent / "coverage.baseline"
NEW_FILE_FLOOR = 80.0
TOLERANCE = 0.005
TOOLS = ("beatcore", "ffmpeg", "onnxruntime")
# Which skipped tests cover which file: tests/test_notes.py and tests/test_analysis.py.
DEPENDS = {
    "app/notes.py": {"ffmpeg", "onnxruntime"},
    "app/analysis.py": {"beatcore", "ffmpeg"},
    "TOTAL": set(TOOLS),
}


def tools_present() -> set[str]:
    present = {"ffmpeg"} if shutil.which("ffmpeg") else set()
    if importlib.util.find_spec("onnxruntime") is not None:
        present.add("onnxruntime")
    # The crate's source folder imports as a namespace package with no origin; the built
    # extension has one.
    beatcore = importlib.util.find_spec("beatcore")
    if beatcore is not None and beatcore.origin is not None:
        present.add("beatcore")
    return present


def sources() -> list[Path]:
    return sorted((ROOT / "app").rglob("*.py"))


def check_fresh(report: Path, files: list[Path]) -> None:
    built = report.stat().st_mtime
    newer = [f for f in files if f.stat().st_mtime > built]
    if newer:
        raise SystemExit(
            f"coverage report is older than {newer[0].name}: run task coverage:run first"
        )


def load_current(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise SystemExit(f"no coverage report at {path}: run task coverage:run first")
    data = json.loads(path.read_text(encoding="utf-8"))
    current = {
        name.replace("\\", "/"): entry["summary"]["percent_covered"]
        for name, entry in data["files"].items()
    }
    current["TOTAL"] = data["totals"]["percent_covered"]
    return current


def read_baseline() -> tuple[dict[str, float], set[str]]:
    accepted: dict[str, float] = {}
    tools: set[str] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# tools:"):
            tools = set(line.removeprefix("# tools:").split())
        elif line and not line.startswith("#"):
            name, _, percent = line.rpartition(" ")
            accepted[name] = float(percent)
    return accepted, tools


def write_baseline(current: dict[str, float], tools: set[str]) -> None:
    header = (
        "# Branch coverage per file, accepted at these percentages.\n"
        "# Regenerate with: task coverage:run, then task coverage:baseline\n"
        f"# tools: {' '.join(sorted(tools))}\n"
    )
    body = "".join(f"{name} {percent:.2f}\n" for name, percent in sorted(current.items()))
    BASELINE.write_text(header + body, encoding="utf-8")


def compare(
    current: dict[str, float],
    baseline: dict[str, float],
    missing: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    found = []
    for name, percent in sorted(current.items()):
        if DEPENDS.get(name, set()) & missing:
            continue
        accepted = baseline.get(name)
        if accepted is None:
            if percent + TOLERANCE < NEW_FILE_FLOOR:
                found.append(f"{name}: {percent:.2f}% < {NEW_FILE_FLOOR:.2f}% floor for a new file")
        elif percent + TOLERANCE < accepted:
            found.append(f"{name}: {percent:.2f}% < baseline {accepted:.2f}%")
    return found


def main() -> int:
    current = load_current(REPORT)
    check_fresh(REPORT, sources())
    tools = tools_present()

    if "--update-baseline" in sys.argv:
        write_baseline(current, tools)
        print(f"baseline written: {BASELINE.relative_to(ROOT)} ({len(current) - 1} file(s))")
        return 0

    if not BASELINE.exists():
        print("no coverage baseline yet: run task coverage:baseline", file=sys.stderr)
        return 1
    baseline, recorded_with = read_baseline()
    missing = recorded_with - tools
    found = compare(current, baseline, missing)
    if found:
        print("\n".join(found), file=sys.stderr)
        return 1
    print(f"coverage {current['TOTAL']:.2f}%, no file below its baseline")
    skipped = sorted(name for name in current if DEPENDS.get(name, set()) & missing)
    if skipped:
        print(f"not compared, this machine lacks {' '.join(sorted(missing))}: {' '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
