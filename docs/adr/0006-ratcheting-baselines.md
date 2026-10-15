# 0006. Quality gates ratchet against a committed baseline

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

The documentation-path and file-length gates come from the toolkit repository, where they
were introduced into a tree that already had findings. Blocking on existing findings helps
nobody; ignoring them lets the count grow.

## Decision

Each gate compares against a committed baseline file. A finding in the baseline passes; a
new one fails. Regenerating the baseline (`task quality:docs:baseline`,
`task quality:long-files:baseline`) is a deliberate act that records progress, never a way to
make a failing build pass.

The one accepted entry at the start is `tasks/claude/budget.py`, vendored verbatim from the
toolkit repository at 328 lines.

## Consequences

**Cost.** A baseline file to keep honest in review.

**Benefit.** The numbers can only go down.

## Alternatives considered

**Hard gates with no baseline.** Fine while there are no findings, and the first legitimate
exception (a long generated file) would force disabling the gate.
