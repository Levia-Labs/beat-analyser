#!/usr/bin/env python3
"""Documentation accuracy gate over the tracked Markdown.

Vendored from the toolkit repository, retargeted. A renamed module, a deleted task or a
moved script leaves a README that still reads correctly and is no longer true, and no linter
notices. Three claims a reader trusts most are checked:

  path    A path a document names resolves on disk, case-exact. The container runs on a
          case-sensitive file system, so a wrong case is a broken instruction - not a typo.
  files   Every script in a folder is named by that folder's README. An unlisted script is
          one nobody documented.
  readme  Every folder holding a script has a README.

Gated: app/, tasks/ and docs/, plus the repository-root documents. Not gated: uploads/,
which is runtime data.

The findings that exist today are recorded in docs.baseline and do not fail the gate. A new
one does, so documentation can only get more accurate.

Usage:
  python3 tasks/quality/docs.py                    # list findings
  python3 tasks/quality/docs.py --check            # fail on a finding not in the baseline
  python3 tasks/quality/docs.py --update-baseline  # accept the current findings
"""

import posixpath
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "docs.baseline"

# Folders whose contents are this repository's own work.
ROOTS = ("app", "tasks", "docs")
ROOT_DOCS = ("README.md", "CLAUDE.md", "TODO.md", ".claude/README.md")
SOURCE_SUFFIXES = (".py",)

SKIP = (
    "graphify-out/",
    "node_modules/",
    # Runtime data: project folders a document names exist only after an upload.
    "uploads/",
)

# A path is only treated as claimed when the token carries a separator and a known suffix,
# or names a folder under a tracked root. Bare basenames stay out: `setup` and `update` are
# task names, and both would otherwise read as paths.
PATH_SUFFIXES = (
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".toml",
    ".sh",
    ".cfg",
    ".conf",
    ".baseline",
    ".desktop",
    ".list",
    ".gpg",
    ".service",
)

# Paths the running system owns. They are real, and absent from a checkout by design: a
# document naming /etc/apt/sources.list.d/vscode.list is describing what a task writes onto
# the workstation, not a file this repository carries.
SYSTEM_ROOTS = (
    "/etc/",
    "/usr/",
    "/var/",
    "/srv/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/opt/",
    "/tmp/",
    "~/",
    "$HOME/",
    "uploads/",
    "graphify-out/",
)

EXEMPT_DOCS: tuple[str, ...] = ()
INLINE_CODE = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# Wildcards, globs and prose fragments are claims about a shape, not about one file.
NOT_A_PATH = re.compile(r"[*?<>{}|\s]|\.\.\.")


def tracked(*patterns: str) -> list[str]:
    # -co --exclude-standard, not a bare ls-files: a document added but not yet
    # committed is precisely the one whose paths have never been checked. The bare
    # form lists only what git already tracks, so a new README was invisible to the
    # gate until after it had been committed.
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", *patterns],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.splitlines()


def on_disk(relative: str) -> bool:
    """Case-exact existence, walked component by component.

    Path.exists() answers case-insensitively on Windows and macOS, which passes a name that
    fails on the Ubuntu machine this repository exists to set up. Walking the components
    also catches the git-ignored per-user files a document may legitimately name.
    """
    current = ROOT
    for part in relative.split("/"):
        if not part or part == ".":
            continue
        try:
            names = {entry.name for entry in current.iterdir()}
        except OSError:
            return False
        if part not in names:
            return False
        current = current / part
    return True


def resolves(claim: str, document: str, index: set[str], folders: set[str]) -> bool:
    """Whether a path a document names points at something real.

    A document names a path from whatever root its reader is standing in: the repository
    root, the folder the README sits in, or a sibling reached with `../`. A repo-relative
    form and a suffix of a tracked path cover all of them without the document having to
    spell out a root it never means.
    """
    claim = claim.rstrip("/")
    if ".." in claim.split("/"):
        claim = posixpath.normpath(posixpath.join(posixpath.dirname(document), claim))
    if claim in index or claim in folders:
        return True
    tail = f"/{claim}"
    if any(path.endswith(tail) for path in index) or any(
        folder.endswith(tail) for folder in folders
    ):
        return True
    return on_disk(claim)


def claimed_paths(text: str) -> set[str]:
    tokens = set(INLINE_CODE.findall(text)) | set(MARKDOWN_LINK.findall(text))
    claims = set()
    for token in tokens:
        token = token.strip().rstrip(".,;:")
        if not token or NOT_A_PATH.search(token) or token.startswith(("http", "#", "mailto:")):
            continue
        if token.startswith(SYSTEM_ROOTS):
            continue
        if (token.endswith(PATH_SUFFIXES) and "/" in token) or token.startswith(ROOTS):
            claims.add(token)
    return claims


def check_paths(document: str, index: set[str], folders: set[str]) -> list[str]:
    text = (ROOT / document).read_text(encoding="utf-8", errors="replace")
    problems = []
    for claim in sorted(claimed_paths(text)):
        if resolves(claim, document, index, folders):
            continue
        problems.append(f"{document}: names '{claim}', which does not resolve (case-exact)")
    return problems


def read_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_baseline(problems: list[str]) -> None:
    header = (
        "# Documentation findings accepted as they stand.\n"
        "# Regenerate with: task quality:docs:baseline\n"
        "# A finding may disappear. Any new one fails task quality:docs:check.\n"
    )
    BASELINE.write_text(header + "".join(f"{p}\n" for p in sorted(problems)), encoding="utf-8")


def collect() -> tuple[list[str], int]:
    in_scope = [p for p in tracked(*ROOTS) if not any(skip in f"{p}" for skip in SKIP)]
    sources = [p for p in in_scope if p.endswith(SOURCE_SUFFIXES)]
    documents = [p for p in in_scope if p.endswith(".md")]
    documents += [p for p in ROOT_DOCS if (ROOT / p).exists()]
    index = set(tracked())
    folders = {Path(p).parent.as_posix() for p in index}

    problems = []
    for document in documents:
        if document in EXEMPT_DOCS:
            continue
        problems.extend(check_paths(document, index, folders))

    for folder in sorted({Path(p).parent.as_posix() for p in sources}):
        readme = ROOT / folder / "README.md"
        if not readme.exists():
            problems.append(f"{folder}: holds a script and has no README.md")
            continue
        text = readme.read_text(encoding="utf-8", errors="replace")
        # A README naming the module is naming the file: `budget.py`, or a code span holding
        # its stem. Only an exact span counts, so a folder heading does not pass for the
        # file that happens to share its name.
        spans = {token.strip().rstrip(".,;:") for token in INLINE_CODE.findall(text)}
        for name in sorted(Path(p).name for p in sources if Path(p).parent.as_posix() == folder):
            if name in text or Path(name).stem in spans:
                continue
            problems.append(f"{folder}/README.md: does not name '{name}'")

    return problems, len(documents)


def main() -> int:
    problems, document_count = collect()

    if "--update-baseline" in sys.argv:
        write_baseline(problems)
        print(f"baseline written: {BASELINE.relative_to(ROOT)} ({len(problems)} findings)")
        return 0

    baseline = read_baseline()
    new = [p for p in problems if p not in baseline]

    if "--check" not in sys.argv:
        for problem in problems:
            print(("NEW  " if problem in new else "     ") + problem)
        print(f"\n{len(problems)} finding(s) across {document_count} document(s), {len(new)} new")
        return 0

    if new:
        print("\n".join(new), file=sys.stderr)
        print(f"\n{len(new)} new documentation finding(s)", file=sys.stderr)
        return 1

    fixed = len(baseline) - (len(problems) - len(new))
    print(f"{document_count} document(s) checked, {len(problems)} known finding(s), none new.")
    if fixed > 0:
        print(f"{fixed} finding(s) fixed; run task quality:docs:baseline to lower the ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
