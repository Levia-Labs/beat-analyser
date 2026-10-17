# 0010. Verification runs locally in two tiers, as tasks

- **Status:** Accepted
- **Date:** 2026-10-17

## Context

Agents verified changes by reading code. Is this import bare? Does the template still avoid
inline script? Is that value ever None? Every such check cost tokens and was only as reliable
as the read. ruff and pytest already ran in `task validate`, which the Stop hook runs after
each edited turn, so a slow or networked check in that gate taxes every turn.

## Decision

Every check is a task, and the checks are split by cost.

`task validate` is offline and takes seconds. It runs ruff, yamllint, basedpyright,
the ast-grep project rules and their tests, pytest with Hypothesis at 50 examples, the
structure gates and the compose check. The Stop hook runs it. Pre-commit and CI run the same
gates as separate steps.

`task validate:deep` runs before a push. It adds a branch-coverage ratchet, Hypothesis at
1000 examples, pip-audit, gitleaks over the whole git history (from its image), and
cargo-audit. The gitleaks pre-commit hook covers staged changes, but it cannot scan history.

The type errors and coverage numbers that predated the gates are baselined and can only
improve ([0006](0006-ratcheting-baselines.md)). The tools are pinned in
`requirements-dev.txt` and run from `.venv`.

## Consequences

**Cost.** `.venv` grows by about 120 MB, most of it the Node runtime basedpyright ships with.
`task validate` takes a few seconds longer. On a new machine, the first commit installs Go for
the gitleaks hook, and the first `rust:audit` builds cargo-audit. Each takes minutes. There
are two more baseline files to keep honest in review.

**Benefit.** A convention written as an ast-grep rule is checked on every turn at no token
cost. Before this, it was checked only when someone remembered to read for it. Type and
property failures show up as one compact line naming file and line, or the smallest input
that fails.

## Alternatives considered

**mypy.** It is slower, and a baseline needs a plugin or a script. basedpyright has one built
in and drops entries once they are fixed.

**ty.** It is faster, but its rule set and configuration were still changing.

**semgrep.** It overlaps ruff's `S` rules and ast-grep, and is a heavy install with
registry-driven rules. ast-grep keeps project rules in the repository, each with a test.

**detect-secrets.** Its baseline file churns on unrelated edits. gitleaks needs no baseline.

**One tier.** Networked audits in the Stop hook would block turns whenever the network is
down, and would put the network's speed into every edit loop.

**Tools installed globally through pipx or uv.** That leaves versions unpinned per repository
and a PATH that depends on the shell, as already happens with ruff from pipx.
