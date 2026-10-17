# tasks / quality

Namespace `quality:`. `docs.py` and `long_files.py` are vendored from the toolkit repository
and retargeted to `app/`, `tasks/`, `docs/` and `tests/`.

| Script | Gate |
| --- | --- |
| `docs.py` | Every path a document names resolves, case-exact; every script is named by its folder's README |
| `long_files.py` | No file past 300 lines |
| `coverage.py` | Per-file branch coverage of `app/` never falls below `coverage.baseline`; a new file needs 80%. Files whose tests need a tool this machine lacks (ffmpeg, onnxruntime, beatcore) are reported, not compared; a report older than the source fails |

Each compares against a committed baseline that ratchets down only
([ADR 0006](../../docs/adr/0006-ratcheting-baselines.md)). No baseline file exists while
there are no findings; `long-files.baseline` holds the vendored `budget.py`.
