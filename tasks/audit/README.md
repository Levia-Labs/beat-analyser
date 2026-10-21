# tasks / audit

Namespace `audit:`. Run by `task validate:deep`, not by `task validate`: it needs the network.

| File           | Task              | Role                                             |
| -------------- | ----------------- | ------------------------------------------------ |
| `Taskfile.yml` | `task audit:venv` | pip-audit over the packages installed in `.venv` |

The shared `security:deps` reads `uv.lock` or a root `requirements.txt`, which this repo does
not pin, so it cannot see `.venv`. `security:secrets` (gitleaks over the history) and
`security:deps` come from the Crepidoma submodule. The Rust dependencies are audited by
`task rust:audit` (cargo-audit in the pinned image).
