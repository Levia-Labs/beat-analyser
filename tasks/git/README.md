# tasks / git

Namespace `git:`. Same names and behaviour as the toolkit repository's `git:` tasks, so one
set of commands works in both. Installing git, its global hardening and GitHub organisation
administration (`git:setup`, `git:verify`, `git:org:*`) stay in the toolkit.

| Task | What it does |
| --- | --- |
| `git:status` | What is uncommitted, and against what upstream |
| `git:update` | Fetch `main` and merge it into the current branch |
| `git:commit [M=...] [SKIP_CHECKS=1]` | Validate, stage, pre-commit, commit, push |

## `git:commit`

1. `task validate`, before staging, so a failure leaves the working tree as it was.
   `SKIP_CHECKS=1` skips it; the git hook still runs on the commit.
2. `git add -A`. Nothing staged is reported as a no-op.
3. `task precommit:staged`, repeated once with the fixers' changes staged.
4. `git commit -m "$M"` and `git push`.

`M` defaults to `update`, with a reminder printed. It pushes, so agents are denied it in
[.claude/settings.json](../../.claude/settings.json); they commit with plain `git commit`.
