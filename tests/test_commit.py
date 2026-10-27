"""The path-scoped commit helper, run against a throwaway git repository."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "dev" / "commit.py"
spec = importlib.util.spec_from_file_location("dev_commit", SCRIPT)
assert spec and spec.loader
commit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commit)


def git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=commit.clean_env(),
    ).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "T")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    git(tmp_path, "add", "seed.txt")
    git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def run(repo, *argv):
    return commit.main(["--repo", str(repo), *argv])


def test_clean_commit(repo, capsys):
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert run(repo, "feat: add a", "a.txt") == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("committed ")
    assert out.endswith(" feat: add a")
    assert git(repo, "log", "-1", "--format=%s").strip() == "feat: add a"


def test_git_env_from_a_hook_is_ignored(repo, monkeypatch):
    monkeypatch.setenv("GIT_DIR", str(repo / "bogus"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(repo / "bogus-index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(repo / "bogus"))
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert run(repo, "feat: a", "a.txt") == 0
    assert git(repo, "log", "-1", "--format=%s").strip() == "feat: a"


def test_coauthor_trailer(repo):
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert run(repo, "feat: a", "a.txt", "--coauthor", "Bob <b@example.com>") == 0
    assert "Co-Authored-By: Bob <b@example.com>" in git(repo, "log", "-1", "--format=%B")


def test_empty_path_list_refused(repo, capsys):
    assert run(repo, "feat: nothing") == 1
    assert "no paths" in capsys.readouterr().out


def test_other_paths_stay_out(repo):
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    assert run(repo, "feat: a", "a.txt") == 0
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["a.txt"]
    assert "?? b.txt" in git(repo, "status", "--short")


def test_rewriting_hook_is_retried(repo, capsys):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "if [ ! -f .git/hook-ran ]; then\n"
        "  : > .git/hook-ran\n"
        "  echo formatted >> a.txt\n"
        "  echo 'reformatted a.txt'\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert run(repo, "feat: a", "a.txt") == 0
    assert git(repo, "show", "HEAD:a.txt") == "a\nformatted\n"


def test_failing_hook_prints_capped_output(repo, capsys):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nseq 1 50\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert run(repo, "feat: a", "a.txt") == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) <= 20
    assert out[-1] == "50"


def test_staged_deletion_is_committed(repo):
    git(repo, "rm", "-q", "seed.txt")
    assert run(repo, "chore: drop seed", "seed.txt") == 0
    assert git(repo, "show", "--name-status", "--format=", "HEAD").split() == ["D", "seed.txt"]


def test_unstaged_deletion_is_committed(repo):
    (repo / "seed.txt").unlink()
    assert run(repo, "chore: drop seed", "seed.txt") == 0
    assert git(repo, "show", "--name-status", "--format=", "HEAD").split() == ["D", "seed.txt"]


@given(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=40))
def test_tail_never_exceeds_cap(lines):
    assert len(commit.tail("\n".join(lines), 20)) <= 20
