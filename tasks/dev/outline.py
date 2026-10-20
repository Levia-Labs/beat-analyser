#!/usr/bin/env python3
"""The shape of a file, with line numbers, so a Read can take only the part it needs.

Reading app/app.py whole to change one route costs 12k characters, kept in context for the
rest of the session; its outline costs a few hundred, and the Read then names offset and
limit (ADR 0011).

  .py        classes, functions and methods (with their end line), UPPER_CASE constants
  .rs        fn, struct, enum, trait, impl, mod, const
  .md        headings
  .yml/.yaml Taskfile tasks (two-space keys under `tasks:`), else top-level keys
  .js        functions and top-level const/let
  .html      elements with an id

Usage:
  python3 tasks/dev/outline.py app/app.py beatcore/src/onset.rs docs/tasks.md
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATTERNS = {
    ".rs": re.compile(
        r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(fn|struct|enum|trait|impl|mod|const)\b\s*"
        r"(?:<[^>]*>\s*)?([A-Za-z_][\w:<>, ]*)"
    ),
    ".md": re.compile(r"^(#{1,4})\s+(.+)"),
    ".js": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(function|const|let)\s+([A-Za-z_$][\w$]*)"),
    ".html": re.compile(r"<(\w+)[^>]*\bid=\"([^\"]+)\""),
}


def python(text: str) -> list[tuple[int, str]]:
    rows = []

    def visit(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                rows.append((node.lineno, f"{kind} {prefix}{node.name}  -{node.end_lineno}"))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{node.name}.")
            elif isinstance(node, ast.Assign) and not prefix:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        rows.append((node.lineno, f"const {target.id}"))

    visit(ast.parse(text).body, "")
    return rows


def yaml(text: str) -> list[tuple[int, str]]:
    rows = []
    section = ""
    for number, line in enumerate(text.splitlines(), 1):
        if top := re.match(r"^([A-Za-z0-9_-]+):", line):
            section = top.group(1)
            rows.append((number, section))
        elif section == "tasks" and (task := re.match(r"^  ([a-z0-9:_-]+):\s*$", line)):
            rows.append((number, f"task {task.group(1)}"))
    return rows


def outline(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".py":
        return python(text)
    if suffix in (".yml", ".yaml"):
        return yaml(text)
    pattern = PATTERNS.get(suffix)
    if pattern is None:
        return []
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in pattern.finditer(line):
            rows.append((number, f"{match.group(1)} {match.group(2).strip()}"))
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    status = 0
    for name in sys.argv[1:]:
        path = (ROOT / name) if not Path(name).is_absolute() else Path(name)
        if not path.is_file():
            print(f"{name}: no such file")
            status = 1
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").count("\n")
        rows = outline(path)
        print(f"{name} ({lines} lines){'' if rows else ': no outline for this type'}")
        for number, label in rows:
            print(f"  {number:5d}  {label}")
    return status


if __name__ == "__main__":
    sys.exit(main())
