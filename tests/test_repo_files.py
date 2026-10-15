"""Guards on tracked text files that a broken encoding breaks far from where it happened.

Windows PowerShell 5.1 writes UTF-16 with a BOM on `>`. A requirements file saved that way
failed the Docker build with `UnicodeDecodeError: 'utf-16-le' codec can't decode`, half an
hour into the build rather than at commit time.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".txt", ".yml", ".yaml", ".md", ".toml", ".ini", ".html", ".js", ".css"}
NAMES = {"Dockerfile", ".dockerignore", ".gitignore", ".gitattributes"}


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


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_utf8_without_bom(path):
    data = path.read_bytes()
    assert not data.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")), "has a BOM"
    data.decode("utf-8")
