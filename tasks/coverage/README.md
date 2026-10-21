# tasks / coverage

Namespace `coverage:`. Per-file branch coverage of `app/`, stricter than the shared
`test:cov`, which ratchets only the total. File-length and documentation gates are the
shared `quality:` namespace from the Crepidoma submodule.

| File                | Task                  | Role                                                                                                                                                                                                                             |
| ------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Taskfile.yml`      | `task coverage:*`     | The `coverage:` namespace                                                                                                                                                                                                        |
| `coverage.py`       | `task coverage:check` | Per-file coverage never falls below `coverage.baseline`; a new file needs 80%. Files whose tests need a tool this machine lacks (ffmpeg, onnxruntime, beatcore) are reported, not compared; a report older than the source fails |
| `coverage.baseline` | -                     | The committed floor; `task coverage:baseline` rewrites it after `task coverage:run`                                                                                                                                              |

The baseline ratchets up only ([ADR 0006](../../docs/adr/0006-ratcheting-baselines.md)).
