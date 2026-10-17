# Local Test and Static-Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add basedpyright, ast-grep, Hypothesis, a coverage ratchet and security scanners to the
go-task pipeline in two tiers, and document how agents use them.

**Architecture:** Every tool is a task. `task validate` (offline, Stop hook, pre-commit) gains
types and ast-grep; `task validate:deep` (manual) adds coverage, deep property runs and networked
audits. Ratchets follow ADR 0006.

**Tech Stack:** go-task 3, basedpyright 1.40.2, ast-grep-cli 0.51.0, coverage 7.16.2,
hypothesis 6.170.1, pip-audit 2.10.1, gitleaks (pre-commit hook), cargo-audit (Rust image).

**Spec:** `docs/superpowers/specs/2026-10-17-local-pipeline-design.md`

## Global Constraints

1. Tools pinned in `requirements-dev.txt`, run from `.venv` (`{{.PY}}` or `{{.BIN}}`).
2. Taskfile shell is `mvdan.cc/sh`, not bash (`.claude/rules/taskfile-shell.md`); included
   Taskfile top-level vars are prefixed with their namespace.
3. `task validate` never touches the network.
4. Baselines ratchet down only (ADR 0006).
5. One command per Bash call; no `cd`; no inline `python -c`.
6. No commits: the user commits. `git add` is fine.
7. No file past 300 lines (`task quality:long-files:check`).

## Review Focus

1. Tool missing from `.venv`: the Stop hook reports "could not run" rather than blocking. Every new
   gate task has a `preconditions` entry, so go-task prints `precondition not met`, which is
   already in `DID_NOT_RUN`.
2. Windows vs Linux venv layout: `.venv/Scripts` vs `.venv/bin`. Root var `BIN` covers both.
3. ast-grep false positives on existing code (`wave.open(buffer, "wb")`, `track.open(newline=...)`):
   rule tests list them as `valid`.
4. `quality:coverage:check` run without a fresh `.cache/coverage.json`: it fails with
   "run task test:cov first", not a traceback. Covered by a test in Task 5.
5. pip-audit offline: `validate:deep` fails with pip-audit's network error; `validate` is
   unaffected.

---

### Task 1: Dependencies, compact output, venv bin var

**Files:**
- Modify: `requirements-dev.txt`, `pytest.ini`, `tasks/lint/Taskfile.yml:19,27`, `.gitignore`, `Taskfile.yml` (vars, doctor)

**Interfaces:**
- Produces: root var `BIN` (venv scripts folder), passed to the `types`, `sg` and `security` includes.

- [ ] **Step 1:** Append to `requirements-dev.txt`:

```
basedpyright==1.40.2
ast-grep-cli==0.51.0
coverage==7.16.2
hypothesis==6.170.1
pip-audit==2.10.1
```

- [ ] **Step 2:** Run `task deps:setup`. Expected: no error.
- [ ] **Step 3:** In `pytest.ini`, change `addopts = -q` to `addopts = -q --tb=short`.
- [ ] **Step 4:** In `tasks/lint/Taskfile.yml`, change both `check` invocations to
  `"{{.RUFF}}" check --output-format concise {{.PY_PATHS}}`.
- [ ] **Step 5:** In `.gitignore` under `# Python`, add `.hypothesis/`.
- [ ] **Step 6:** In root `Taskfile.yml` vars, add:

```yaml
  # The virtualenv's scripts folder, for tools that ship a binary (ast-grep).
  BIN: '{{if eq OS "windows"}}.venv/Scripts{{else}}.venv/bin{{end}}'
```

  In `doctor`, after the `imports ok` line:

```yaml
          "{{.PY}}" -c "import basedpyright, coverage, hypothesis, pip_audit; print('  devtools  ok (basedpyright, coverage, hypothesis, pip-audit)')" \
            || echo "  devtools  MISSING - run task deps:setup"
```

- [ ] **Step 7:** Run `task doctor` and `task test`. Expected: `devtools ok`, tests pass.

### Task 2: basedpyright

**Files:**
- Create: `pyrightconfig.json`, `tasks/types/Taskfile.yml`, `tasks/types/README.md`, `.basedpyright/baseline.json` (generated)
- Modify: `Taskfile.yml` (includes, validate), `.pre-commit-config.yaml`

- [ ] **Step 1:** Create `pyrightconfig.json`:

```json
{
  "include": ["app", "tasks", "tests"],
  "exclude": ["**/.ruff_cache", "**/__pycache__", ".venv", "graphify-out", "uploads"],
  "venvPath": ".",
  "venv": ".venv",
  "pythonVersion": "3.14",
  "typeCheckingMode": "standard",
  "executionEnvironments": [
    { "root": "tasks", "pythonVersion": "3.12" },
    { "root": "tests", "extraPaths": ["app", "tests"] },
    { "root": "app", "extraPaths": ["app"] }
  ]
}
```

- [ ] **Step 2:** Create `tasks/types/Taskfile.yml`:

```yaml
version: "3"

# basedpyright over app/, tasks/ and tests/ (pyrightconfig.json). Errors that predate the
# checker live in .basedpyright/baseline.json; basedpyright drops an entry once it is fixed,
# so the count only falls (docs/adr/0006).

tasks:
  check:
    desc: Type check. Fails on an error not in the baseline. Changes nothing else.
    preconditions:
      - sh: '"{{.PY}}" -c "import basedpyright"'
        msg: basedpyright is not in .venv - run task deps:setup
    cmds:
      - '"{{.PY}}" -m basedpyright'

  baseline:
    desc: Accept the current type errors as the ceiling
    cmds:
      - '"{{.PY}}" -m basedpyright --writebaseline'
```

  `tasks/types/README.md`:

```markdown
# tasks / types

Namespace `types:`. basedpyright, configured in [pyrightconfig.json](../../pyrightconfig.json):
`app/` and `tests/` on Python 3.14, `tasks/` on 3.12. The baseline
`.basedpyright/baseline.json` ratchets down only ([ADR 0006](../../docs/adr/0006-ratcheting-baselines.md)).
```

- [ ] **Step 3:** Root `Taskfile.yml` includes: `types: { taskfile: ./tasks/types/Taskfile.yml, vars: { PY: "{{.PY}}" } }`.
- [ ] **Step 4:** Run `task types:check`. Note the error count. Fix errors whose fix is one obvious
  line; leave the rest.
- [ ] **Step 5:** Run `task types:baseline`, then `task types:check`. Expected: `0 errors`.
- [ ] **Step 6:** Add `- task: types:check` after `lint:check` in `validate`. Add a pre-commit
  local hook `{id: types, name: basedpyright, entry: task types:check, language: system,
  pass_filenames: false, files: \.py$|^pyrightconfig\.json$}`.

### Task 3: ast-grep rules

**Files:**
- Create: `sgconfig.yml`, `tasks/sg/Taskfile.yml`, `tasks/sg/README.md`, `tasks/sg/rules/*.yml` (5), `tasks/sg/tests/*-test.yml` (5)
- Modify: `Taskfile.yml` (includes, validate), `.pre-commit-config.yaml`, any source a rule flags

- [ ] **Step 1:** Create `sgconfig.yml`:

```yaml
# ast-grep: project rules in tasks/sg/rules, one test file each in tasks/sg/tests.
ruleDirs:
  - tasks/sg/rules
testConfigs:
  - testDir: tasks/sg/tests
```

- [ ] **Step 2: Write rule tests first.** `tasks/sg/tests/yt-dlp-only-in-youtube-test.yml`:

```yaml
id: yt-dlp-only-in-youtube
valid:
  - "download_audio(vid, folder, 60, 10)"
invalid:
  - "with YoutubeDL(options) as ydl: pass"
```

  `app-bare-imports-test.yml`:

```yaml
id: app-bare-imports
valid:
  - "import storage"
  - "from storage import is_valid_name"
invalid:
  - "from . import storage"
  - "from .storage import is_valid_name"
  - "from app import storage"
```

  `text-io-encoding-test.yml`:

```yaml
id: text-io-encoding
valid:
  - 'path.read_text(encoding="utf-8")'
  - 'path.write_text(body, encoding="utf-8")'
  - 'open(path, "rb")'
  - 'wave.open(buffer, "wb")'
  - 'track.open(newline="", encoding="utf-8")'
  - "path.read_bytes()"
invalid:
  - "path.read_text()"
  - "path.write_text(body)"
  - "open(path)"
  - 'open(path, "w")'
  - 'path.open("w")'
```

  `no-print-in-app-test.yml`:

```yaml
id: no-print-in-app
valid:
  - 'log.info("done")'
invalid:
  - 'print("done")'
```

  `template-no-inline-test.yml`:

```yaml
id: template-no-inline
valid:
  - '<script src="/static/app.js"></script>'
  - '<div class="page"></div>'
invalid:
  - "<script>alert(1)</script>"
  - '<div style="color: red"></div>'
```

- [ ] **Step 3:** Run `.venv/Scripts/ast-grep test --skip-snapshot-tests` (via `task sg:test` once
  Step 5 exists). Expected: FAIL, rules not found.
- [ ] **Step 4: Write rules.** `tasks/sg/rules/yt-dlp-only-in-youtube.yml`:

```yaml
id: yt-dlp-only-in-youtube
language: python
severity: error
message: yt-dlp is called only from app/youtube.py, after youtube.video_id reduced the URL.
rule:
  pattern: YoutubeDL($$$ARGS)
ignores:
  - app/youtube.py
```

  `app-bare-imports.yml`:

```yaml
id: app-bare-imports
language: python
severity: error
message: App modules import each other by bare name (`import storage`); gunicorn loads them from /app.
files:
  - app/**/*.py
rule:
  kind: import_from_statement
  has:
    field: module_name
    any:
      - kind: relative_import
      - regex: '^app(\.|$)'
```

  `text-io-encoding.yml`:

```yaml
id: text-io-encoding
language: python
severity: error
message: Pass encoding="utf-8". The Windows default is cp1252, and a BOM or mojibake breaks far from here.
rule:
  any:
    - pattern: $P.read_text()
    - pattern: $P.write_text($TEXT)
    - pattern: open($FILE)
    - pattern: open($FILE, $MODE)
    - pattern: $P.open($MODE)
constraints:
  MODE:
    regex: '^["''][^b"'']*["'']$'
```

  `no-print-in-app.yml`:

```yaml
id: no-print-in-app
language: python
severity: error
message: The service logs; only app/cli.py prints.
files:
  - app/**/*.py
ignores:
  - app/cli.py
rule:
  pattern: print($$$ARGS)
```

  `template-no-inline.yml`:

```yaml
id: template-no-inline
language: html
severity: error
message: The CSP forbids inline script and style; move it to app/static/.
files:
  - app/templates/**/*.html
rule:
  any:
    - kind: attribute_name
      regex: '^style$'
    - all:
        - kind: script_element
        - not:
            has:
              kind: attribute_name
              regex: '^src$'
              stopBy: end
```

- [ ] **Step 5:** Create `tasks/sg/Taskfile.yml`:

```yaml
version: "3"

# ast-grep: structural search and the project rules in tasks/sg/rules (sgconfig.yml).
# A rule turns a CLAUDE.md convention into a check; each has a test in tasks/sg/tests.

vars:
  SG_BIN: '{{.BIN}}/ast-grep'

tasks:
  check:
    desc: Run the project rules over the tree. Changes nothing.
    preconditions:
      - sh: '"{{.SG_BIN}}" --version'
        msg: ast-grep is not in .venv - run task deps:setup
    cmds:
      - '"{{.SG_BIN}}" scan --report-style short'

  test:
    desc: Test every rule against its valid and invalid snippets
    preconditions:
      - sh: '"{{.SG_BIN}}" --version'
        msg: ast-grep is not in .venv - run task deps:setup
    cmds:
      - '"{{.SG_BIN}}" test --skip-snapshot-tests'

  search:
    desc: 'Structural search. Usage: task sg:search P="subprocess.run($$$A)" [L=python] [-- app]'
    requires:
      vars: [P]
    cmds:
      - '"{{.SG_BIN}}" run --pattern "{{.P}}" --lang {{.L | default "python"}} {{.CLI_ARGS}}'
```

  `tasks/sg/README.md`:

```markdown
# tasks / sg

Namespace `sg:`. [ast-grep](https://ast-grep.github.io/) rules in `rules/`, one test file per
rule in `tests/`, wired by [sgconfig.yml](../../sgconfig.yml). A new rule needs a test listing
at least one `valid` and one `invalid` snippet; `task sg:test` runs them.
```

  Root includes: `sg: { taskfile: ./tasks/sg/Taskfile.yml, vars: { BIN: "{{.BIN}}" } }`.
- [ ] **Step 6:** Run `task sg:test`. Expected: all 5 pass. Fix a rule, never a test, until green.
- [ ] **Step 7:** Run `task sg:check`. Fix every finding in source (add `encoding="utf-8"`).
  Expected: exit 0.
- [ ] **Step 8:** Add `- task: sg:check` and `- task: sg:test` to `validate` after
  `types:check`. Add pre-commit local hook `{id: ast-grep, name: ast-grep rules, entry: task
  sg:check, language: system, pass_filenames: false, files: \.(py|html)$|^tasks/sg/|^sgconfig\.yml$}`.

### Task 4: Hypothesis properties

**Files:**
- Modify: `tests/conftest.py`, `Taskfile.yml` (test:deep)
- Create: `tests/test_properties.py`

- [ ] **Step 1:** In `tests/conftest.py`, after the imports add `import os` (sorted) and
  `from hypothesis import settings`, then:

```python
# Fast in task test and the Stop hook; task test:deep sets HYPOTHESIS_PROFILE=deep.
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("deep", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
```

- [ ] **Step 2:** Create `tests/test_properties.py`:

```python
"""Properties of the pure functions at the trust boundary and in the CSV writer.

Example tests pin known cases; these search for the inputs nobody thought of. Profiles are
in conftest.py: `task test` runs 50 examples, `task test:deep` 1000.
"""

import string

from hypothesis import given
from hypothesis import strategies as st

import analysis
import notes
import storage
import youtube

ID = st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=11, max_size=11)


@given(st.text())
def test_project_name_is_valid_or_refused(filename):
    try:
        name = storage.project_name(filename)
    except ValueError:
        return
    assert storage.is_valid_name(name)
    assert "/" not in name and "\\" not in name


@given(st.text())
def test_project_name_is_stable(filename):
    try:
        name = storage.project_name(filename)
    except ValueError:
        return
    assert storage.project_name(f"{name}.wav") == name


@given(st.text())
def test_valid_name_cannot_traverse(name):
    if storage.is_valid_name(name):
        assert not name.startswith(".")
        assert "/" not in name and "\\" not in name
        assert len(name) <= 100


@given(st.text())
def test_video_id_is_an_id_or_refused(url):
    try:
        vid = youtube.video_id(url)
    except ValueError:
        return
    assert len(vid) == 11
    assert set(vid) <= set(string.ascii_letters + string.digits + "_-")


@given(ID)
def test_video_id_round_trips(vid):
    assert youtube.video_id(youtube.canonical_url(vid)) == vid
    assert youtube.video_id(f"https://youtu.be/{vid}") == vid


@given(st.floats(0, 1e5), st.floats(0, 1e5))
def test_format_time_is_monotonic_and_close(a, b):
    low, high = sorted((a, b))
    assert float(analysis.format_time(low)) <= float(analysis.format_time(high))
    assert abs(float(analysis.format_time(a)) - a) <= 5e-5


NOTE = st.tuples(
    st.floats(0, 100), st.floats(0, 5), st.integers(21, 108), st.floats(0, 1)
).map(lambda t: (t[0], t[0] + t[1], t[2], t[3]))


@given(st.lists(NOTE, max_size=40))
def test_merge_repeats_shortens_sorts_and_separates(found):
    merged = notes.merge_repeats(found)
    assert len(merged) <= len(found)
    assert merged == sorted(merged)
    last_end: dict[int, float] = {}
    for start, end, pitch, _ in merged:
        if pitch in last_end:
            assert start - last_end[pitch] >= notes.MERGE_GAP_S
        last_end[pitch] = max(end, last_end.get(pitch, end))
```

- [ ] **Step 3:** Run `task test K=properties`. A failure is a real finding: report the
  falsifying example, then fix the code or narrow the property only where the example is
  outside the function's contract (say which in the report).
- [ ] **Step 4:** Add to root `Taskfile.yml`:

```yaml
  test:deep:
    desc: Property tests at 1000 examples each (Hypothesis deep profile)
    env:
      HYPOTHESIS_PROFILE: deep
    cmds:
      - '"{{.PY}}" -m pytest tests/test_properties.py {{.CLI_ARGS}}'
```

- [ ] **Step 5:** Run `task test:deep`. Expected: pass.

### Task 5: Coverage ratchet

**Files:**
- Create: `.coveragerc`, `tasks/quality/coverage.py`, `tasks/quality/coverage.baseline` (generated), `tests/test_coverage_gate.py`
- Modify: `tasks/quality/Taskfile.yml`, `tasks/quality/README.md`, `Taskfile.yml` (test:cov)

**Interfaces:**
- Produces: `coverage.py` functions `compare(current: dict[str, float], baseline: dict[str, float]) -> list[str]` and `load_current(path: Path) -> dict[str, float]`; key `"TOTAL"` holds the total.

- [ ] **Step 1: Failing test.** `tests/test_coverage_gate.py`:

```python
"""The coverage ratchet: a file may hold or gain coverage, never lose it."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "quality" / "coverage.py"
spec = importlib.util.spec_from_file_location("coverage_gate", SCRIPT)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_holding_or_gaining_passes():
    assert gate.compare({"TOTAL": 80.0, "app/a.py": 90.0}, {"TOTAL": 80.0, "app/a.py": 85.0}) == []


def test_a_drop_fails():
    found = gate.compare({"TOTAL": 80.0, "app/a.py": 84.0}, {"TOTAL": 80.0, "app/a.py": 85.0})
    assert found == ["app/a.py: 84.0% < baseline 85.0%"]


def test_a_new_file_under_the_floor_fails():
    found = gate.compare({"TOTAL": 80.0, "app/new.py": 10.0}, {"TOTAL": 80.0})
    assert found == [f"app/new.py: 10.0% < {gate.NEW_FILE_FLOOR:.1f}% floor for a new file"]


def test_rounding_noise_passes():
    assert gate.compare({"TOTAL": 79.96}, {"TOTAL": 80.0}) == []


def test_missing_report_says_what_to_run(tmp_path):
    with pytest.raises(SystemExit, match="task test:cov"):
        gate.load_current(tmp_path / "absent.json")
```

- [ ] **Step 2:** Run `task test K=coverage_gate`. Expected: FAIL, script missing.
- [ ] **Step 3:** Create `tasks/quality/coverage.py`:

```python
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
                found.append(
                    f"{name}: {percent:.1f}% < {NEW_FILE_FLOOR:.1f}% floor for a new file"
                )
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
```

- [ ] **Step 4:** Run `task test K=coverage_gate`. Expected: 5 pass.
- [ ] **Step 5:** Create `.coveragerc`:

```ini
# task test:cov. Branch coverage of the service; tasks/ and tests/ are not measured.
[run]
branch = True
source = app

[report]
skip_covered = True
show_missing = True
precision = 1

[json]
output = .cache/coverage.json
```

  Root `Taskfile.yml`:

```yaml
  test:cov:
    desc: Tests under branch coverage; lists only files below 100% and their missing lines
    cmds:
      - '"{{.PY}}" -m coverage run -m pytest {{.CLI_ARGS}}'
      - '"{{.PY}}" -m coverage report'
      - '"{{.PY}}" -m coverage json -q'
```

  `tasks/quality/Taskfile.yml`, after `long-files:baseline`:

```yaml
  coverage:check:
    desc: Fail when a file's coverage falls below the baseline (run task test:cov first)
    cmds:
      - '"{{.SYS_PY}}" tasks/quality/coverage.py --check'

  coverage:baseline:
    desc: Accept the current coverage as the floor (run task test:cov first)
    cmds:
      - '"{{.SYS_PY}}" tasks/quality/coverage.py --update-baseline'
```

  `tasks/quality/README.md`: add row `| coverage.py | Per-file branch coverage of app/ never falls below its baseline; a new file needs 80% |`
  and change "Both gates" to "The gates".
- [ ] **Step 6:** Run `task test:cov`, `task quality:coverage:baseline`, `task quality:coverage:check`. Expected: last prints `no file below its baseline`.

### Task 6: Security scanners

**Files:**
- Create: `tasks/security/Taskfile.yml`, `tasks/security/README.md`
- Modify: `.pre-commit-config.yaml`, `tasks/rust/Taskfile.yml`, `Taskfile.yml` (includes)

- [ ] **Step 1:** Add to `.pre-commit-config.yaml` before `pre-commit-hooks`:

```yaml
  # Secrets in staged changes. pre-commit installs the Go toolchain the hook needs.
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.28.0
    hooks:
      - id: gitleaks
```

  Run `pre-commit autoupdate --repo https://github.com/gitleaks/gitleaks` to pin the latest tag.
  If the golang hook cannot install on Windows, switch the id to `gitleaks-docker` and say so.
- [ ] **Step 2:** Create `tasks/security/Taskfile.yml`:

```yaml
version: "3"

# The networked and slow security checks, run by task validate:deep. ruff's S rules (bandit)
# and the gitleaks pre-commit hook cover the fast path.

vars:
  SECURITY_PRECOMMIT:
    sh: command -v pre-commit 2>/dev/null || printf '%s' "${PIPX_BIN_DIR:-$HOME/.local/bin}/pre-commit"

tasks:
  deps:
    desc: Known vulnerabilities in the packages installed in .venv (pip-audit; needs network)
    preconditions:
      - sh: '"{{.PY}}" -c "import pip_audit"'
        msg: pip-audit is not in .venv - run task deps:setup
    cmds:
      - '"{{.PY}}" -m pip_audit --progress-spinner off --skip-editable'

  secrets:
    desc: Secrets in every tracked file (gitleaks through pre-commit)
    cmds:
      - '"{{.SECURITY_PRECOMMIT}}" run gitleaks --all-files'
```

  `tasks/security/README.md`:

```markdown
# tasks / security

Namespace `security:`. Run by `task validate:deep`, not by `task validate`: both need the
network or a first-run install.

| Task | Tool |
| --- | --- |
| `security:deps` | pip-audit over the packages installed in `.venv` |
| `security:secrets` | gitleaks over every tracked file, through its pre-commit hook |

The Rust dependencies are audited by `task rust:audit` (cargo-audit in the pinned image).
```

  Root includes: `security: { taskfile: ./tasks/security/Taskfile.yml, vars: { PY: "{{.PY}}" } }`.
- [ ] **Step 3:** Add to `tasks/rust/Taskfile.yml`:

```yaml
  audit:
    desc: RustSec advisories for beatcore/Cargo.lock (cargo-audit in the pinned image; needs network)
    cmds:
      - >-
        docker run --rm
        -v "{{.ROOT_DIR}}/beatcore:/src"
        -v beatcore-cargo:/usr/local/cargo/registry
        -v beatcore-tools:/opt/tools
        -v beatcore-advisory:/usr/local/cargo/advisory-db
        -w /src {{.RUST_IMAGE}}
        sh -c "[ -x /opt/tools/bin/cargo-audit ] || cargo install --quiet --locked --root /opt/tools cargo-audit; /opt/tools/bin/cargo-audit audit"
```

- [ ] **Step 4:** Run `task security:deps`, `task security:secrets`, `task rust:audit`. Record
  each result. A finding is reported, not silenced.

### Task 7: validate:deep and the Stop hook

**Files:**
- Modify: `Taskfile.yml`, `tasks/claude/hooks.py:43-49`

- [ ] **Step 1:** Root `Taskfile.yml`, after `validate`:

```yaml
  validate:deep:
    desc: validate, then coverage ratchet, deep property tests and security audits. Before a push; needs network.
    cmds:
      - task: validate
      - task: test:cov
      - task: quality:coverage:check
      - task: test:deep
      - task: security:deps
      - task: security:secrets
      - task: rust:audit
```

  Update the `validate` desc to `Every offline gate - task tree, lint, types, ast-grep, tests, structure, compose file. The Stop hook and pre-commit run it.`
  Update the header comment's EVERY DAY block with `task validate:deep`.
- [ ] **Step 2:** In `hooks.py` `DID_NOT_RUN`, add `"No module named basedpyright"`.
- [ ] **Step 3:** Run `task validate`. Expected: exit 0. Time it; report if over 30 s.

### Task 8: Docs, CLAUDE.md, ADR

**Files:**
- Create: `docs/adr/0010-two-tier-local-pipeline.md`
- Modify: `CLAUDE.md`, `docs/adr/README.md`, `docs/tasks.md`, `docs/development.md`, `tasks/README.md`

- [ ] **Step 1:** ADR 0010 from `docs/adr/template.md`: Decision = two tiers, tools as listed,
  ratchets. Alternatives = mypy, ty, semgrep, detect-secrets, one tier, global pipx tools.
  Consequences = ~120 MB more in `.venv` (node for basedpyright), first `security:secrets`
  run installs Go, validate slower by the type check. Index row in `docs/adr/README.md`.
- [ ] **Step 2:** `tasks/README.md`: rows for `types/`, `sg/`, `security/`; `quality/` row adds coverage.
- [ ] **Step 3:** `docs/tasks.md` and `docs/development.md`: entries for every new task, in the
  existing format of each file.
- [ ] **Step 4:** `CLAUDE.md`: in "Verifying a change" add `task types:check`, `task sg:check`,
  `task validate:deep`; add a section "Finding things cheaply": `task graphify:query` for
  relationships, `task sg:search P=` for code shape, Grep for literals, never a read sweep;
  rtk rewrites raw shell commands, tasks already print compact output, `rtk gain` measures.
  Add to the "Instead of" table: `a grep for a call shape` -> `task sg:search P='requests.get($$$)'`;
  `reading code to check a convention` -> `a rule in tasks/sg/rules`. Add to Conventions: a
  convention that can be a rule becomes an ast-grep rule with a test.
- [ ] **Step 5:** Run `humanizer` skill over ADR 0010. Run `task validate`. Expected: exit 0.
