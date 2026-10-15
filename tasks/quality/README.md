# tasks / quality

Namespace `quality:`. Both gates are vendored from the toolkit repository and retargeted to
`app/`, `tasks/`, `docs/` and `tests/`.

| Script | Gate |
| --- | --- |
| `docs.py` | Every path a document names resolves, case-exact; every script is named by its folder's README |
| `long_files.py` | No file past 300 lines |

Each compares against a committed baseline that ratchets down only
([ADR 0006](../../docs/adr/0006-ratcheting-baselines.md)). No baseline file exists while
there are no findings; `long-files.baseline` holds the vendored `budget.py`.
