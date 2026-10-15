#!/usr/bin/env python3
"""Find, and optionally fix, tracked text files that are not UTF-8 without a BOM.

app/requirements.txt was committed as UTF-16 (a Windows PowerShell `>` redirect) and broke
the Docker build. The Write tool preserves a file's existing encoding, so rewriting such a
file does not fix it; this does.

Usage:
  python3 tasks/dev/encoding.py           # list offenders, exit 1 if any
  python3 tasks/dev/encoding.py --fix     # rewrite them as UTF-8, no BOM, LF endings
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".py", ".txt", ".yml", ".yaml", ".md", ".toml", ".ini", ".html", ".js", ".css"}
NAMES = {"Dockerfile", ".dockerignore", ".gitignore", ".gitattributes"}
BOMS = {b"\xff\xfe": "utf-16-le", b"\xfe\xff": "utf-16-be", b"\xef\xbb\xbf": "utf-8-sig"}


def text_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [
        ROOT / name
        for name in listed
        if (Path(name).suffix in SUFFIXES or Path(name).name in NAMES)
        and not name.startswith(".claude/skills/")
        and (ROOT / name).is_file()
    ]


def detect(data: bytes) -> str | None:
    """The encoding to decode with when the file is NOT clean UTF-8, else None."""
    for bom, codec in BOMS.items():
        if data.startswith(bom):
            return codec
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "cp1252"
    return None


def main() -> int:
    fix = "--fix" in sys.argv
    offenders = 0
    for path in text_files():
        data = path.read_bytes()
        codec = detect(data)
        if codec is None:
            continue
        offenders += 1
        relative = path.relative_to(ROOT).as_posix()
        if not fix:
            print(f"  {relative}: {codec}")
            continue
        if codec.startswith("utf-16") and len(data) % 2:
            data = data[:-1]  # a truncated final byte; the character it held is a newline
        text = data.decode(codec, errors="replace").replace("\r\n", "\n")
        if not text.endswith("\n"):
            text += "\n"
        path.write_bytes(text.encode("utf-8"))
        print(f"  fixed {relative} ({codec} -> utf-8)")
    if offenders and not fix:
        print(f"{offenders} file(s) not UTF-8. Fix: task dev:encoding:fix")
        return 1
    if not offenders:
        print("every tracked text file is UTF-8 without a BOM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
