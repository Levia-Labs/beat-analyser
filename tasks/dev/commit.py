"""Commit exactly the given paths, retrying once when hooks reformatted them.

Commands:
    commit.py "<message>" <path> [<path>...] [--coauthor "Name <email>"]

Stages the paths, runs `git commit -m <message> -- <paths>` so nothing else is committed,
and on failure re-stages any of those paths the hooks rewrote (prettier, ruff format) and
retries once. Prints `committed <short-hash> <subject>` on success; on failure at most 20
lines of hook output and exit 1. Never pushes, never uses --no-verify. Exit 2 on usage errors.

Usage: task dev:commit COAUTHOR="Name <email>" -- "feat(x): msg" path1 path2
"""

import argparse
import os
import subprocess
import sys

CAP = 20


def clean_env() -> dict[str, str]:
    """The environment without GIT_*: inside a git hook GIT_DIR and GIT_INDEX_FILE point at
    the outer repository and would redirect every call here."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=clean_env(),
    )


def tail(text: str, cap: int = CAP) -> list[str]:
    """The last `cap` non-empty lines: hook verdicts print at the end."""
    return [line for line in text.splitlines() if line.strip()][-cap:]


def commit_args(message: str, coauthor: str, paths: list[str]) -> list[str]:
    args = ["commit", "-m", message]
    if coauthor:
        args += ["-m", f"Co-Authored-By: {coauthor}"]
    return [*args, "--", *paths]


def rewritten(repo: str, paths: list[str]) -> list[str]:
    """Given paths whose working tree differs from the index, i.e. a hook rewrote them."""
    return git(repo, "diff", "--name-only", "--", *paths).stdout.split()


def run(repo: str, message: str, paths: list[str], coauthor: str) -> tuple[int, list[str]]:
    if not paths:
        return 1, ["no paths given"]
    added = git(repo, "add", "--", *paths)
    if added.returncode:
        return 1, tail(added.stderr + added.stdout)
    args = commit_args(message, coauthor, paths)
    done = git(repo, *args)
    if done.returncode:
        changed = rewritten(repo, paths)
        if changed:
            git(repo, "add", "--", *changed)
            done = git(repo, *args)
    if done.returncode:
        return 1, tail(done.stdout + "\n" + done.stderr)
    head = git(repo, "log", "-1", "--format=%h %s").stdout.strip()
    return 0, [f"committed {head}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("message")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--coauthor", default="")
    parser.add_argument("--repo", default=".", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    status, lines = run(args.repo, args.message, args.paths, args.coauthor)
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
