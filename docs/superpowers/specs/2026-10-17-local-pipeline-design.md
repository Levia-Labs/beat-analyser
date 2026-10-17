# Local test and static-analysis pipeline

- **Date:** 2026-10-17
- **Status:** Approved design

## Goal

Move verification work from the LLM to the local machine. Every check an agent would otherwise
do by reading code becomes a task with compact output. Fast checks gate every turn; slow and
networked checks run on demand.

## Decisions (from the user)

1. Type checker: basedpyright.
2. Two tiers: `task validate` (offline, fast, Stop hook and pre-commit) and `task validate:deep`
   (manual, before a push).
3. New Python tools pinned in `requirements-dev.txt`, installed into `.venv`.
4. Coverage ratchets against a committed per-file baseline (ADR 0006).

## Tiers

`task validate`: task tree, `lint:check`, `types:check`, `sg:check`, `sg:test`, `test`
(Hypothesis `dev` profile), `quality:check`, `docker:config`, `claude:allow:report`. No network.

`task validate:deep`: `validate`, `test:cov` with `quality:coverage:check`, `test:deep`
(Hypothesis `deep` profile), `security:deps`, `security:secrets`, `rust:audit`.

## Components

| Tool | Where | Tasks | Gate |
| --- | --- | --- | --- |
| basedpyright | `.venv`, `pyrightconfig.json`, `.basedpyright/baseline.json` | `types:check`, `types:baseline` | validate |
| ast-grep | `.venv` (`ast-grep-cli`), `sgconfig.yml`, `tasks/sg/rules`, `tasks/sg/tests` | `sg:check`, `sg:test`, `sg:search` | validate |
| coverage.py | `.venv`, `.coveragerc`, `tasks/quality/coverage.py`, `tasks/quality/coverage.baseline` | `test:cov`, `quality:coverage:check`, `quality:coverage:baseline` | deep |
| Hypothesis | `.venv`, profiles in `tests/conftest.py` | `test` (dev), `test:deep` | validate, deep |
| pip-audit | `.venv` | `security:deps` | deep |
| gitleaks | pre-commit hook | `security:secrets` | commit, deep |
| cargo-audit | pinned Rust image, binary in volume `beatcore-cargo-bin` | `rust:audit` | deep |

basedpyright environments: `app/` and `tests/` on Python 3.14 with `app` on the import path;
`tasks/` on 3.12. Mode `standard`. Existing errors live in the baseline; basedpyright prunes
fixed entries, so the count only falls.

ast-grep rules (each with a valid and an invalid case under `tasks/sg/tests/`):

1. `yt-dlp-only-in-youtube`: `YoutubeDL(...)` outside `app/youtube.py`.
2. `app-bare-imports`: relative imports or `from app import` under `app/`.
3. `text-io-encoding`: `read_text()`, `write_text(x)` without `encoding=`.
4. `no-print-in-app`: `print(` under `app/` except `app/cli.py`.
5. `template-no-inline`: inline `<script>` without `src`, any `style=` attribute.

Property tests (`tests/test_properties.py`):

1. `storage.project_name` output is valid by `is_valid_name` and stable under repetition.
2. `youtube.video_id` returns an 11-character `[A-Za-z0-9_-]` ID or raises `ValueError`.
3. `analysis.format_time` is monotonic.
4. `notes.merge_repeats` never lengthens its input and keeps it sorted.

## Token economy

1. Tasks print compact output: `ruff --output-format concise`, `pytest -q --tb=short`, coverage
   lists only files below 100 percent with missing lines.
2. RTK's global hook rewrites raw shell commands; tasks are preferred anyway.
3. Search order in CLAUDE.md: `graphify:query` for relationships, `sg:search` for code shape,
   Grep for literals.

## Error handling

A tool missing from `.venv` is reported by the Stop hook as "could not run", not as a failure
(`DID_NOT_RUN` in `tasks/claude/hooks.py`). Network checks never run in `validate`.

## Docs

`CLAUDE.md`, `docs/tasks.md`, `docs/development.md`, `tasks/README.md`, ADR 0010, ADR index.

## Verification

`task validate` green; `task validate:deep` run once, result reported; `task sg:test` passes.

## Out of scope

CI workflow, semgrep, hadolint, mypy, git-history secret scan.
