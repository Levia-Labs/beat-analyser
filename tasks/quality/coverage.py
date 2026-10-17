#!/usr/bin/env python3
"""Coverage ratchet over app/, from the JSON report task test:cov writes.

Per file and in total, coverage may hold or rise, never fall below the committed baseline. A
file the baseline has never seen must reach NEW_FILE_FLOOR. Tolerance absorbs the rounding of
a one-line change in a large file, not a real drop.

Usage:
  python3 tasks/quality/coverage.py --check            # fail on a drop
  python3 tasks/quality/coverage.py --update-baseline  # accept the current numbers
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / ".cache" / "coverage.json"
BASELINE = Path(__file__).resolve().parent / "coverage.baseline"
NEW_FILE_FLOOR = 80.0
TOLERANCE = 0.05


def load_current(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise SystemExit(f"no coverage report at {path}: run task test:cov first")
    data = json.loads(path.read_text(encoding="utf-8"))
    current = {
        name.replace("\\", "/"): entry["summary"]["percent_covered"]
        for name, entry in data["files"].items()
    }
    current["TOTAL"] = data["totals"]["percent_covered"]
    return current


def read_baseline() -> dict[str, float]:
    if not BASELINE.exists():
        return {}
    accepted = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, percent = line.rpartition(" ")
            accepted[name] = float(percent)
    return accepted


def write_baseline(current: dict[str, float]) -> None:
    header = (
        "# Branch coverage per file, accepted at these percentages.\n"
        "# Regenerate with: task test:cov, then task quality:coverage:baseline\n"
    )
    body = "".join(f"{name} {percent:.1f}\n" for name, percent in sorted(current.items()))
    BASELINE.write_text(header + body, encoding="utf-8")


def compare(current: dict[str, float], baseline: dict[str, float]) -> list[str]:
    found = []
    for name, percent in sorted(current.items()):
        accepted = baseline.get(name)
        if accepted is None:
            if percent + TOLERANCE < NEW_FILE_FLOOR:
                found.append(f"{name}: {percent:.1f}% < {NEW_FILE_FLOOR:.1f}% floor for a new file")
        elif percent + TOLERANCE < accepted:
            found.append(f"{name}: {percent:.1f}% < baseline {accepted:.1f}%")
    return found


def main() -> int:
    current = load_current(REPORT)
    if "--update-baseline" in sys.argv:
        write_baseline(current)
        print(f"baseline written: {BASELINE.relative_to(ROOT)} ({len(current) - 1} file(s))")
        return 0
    found = compare(current, read_baseline())
    if found:
        print("\n".join(found), file=sys.stderr)
        return 1
    print(f"coverage {current['TOTAL']:.1f}%, no file below its baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
