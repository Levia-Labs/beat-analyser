# tasks / security

Namespace `security:`. Run by `task validate:deep`, not by `task validate`: both need the
network or a first-run install.

| Task | Tool |
| --- | --- |
| `security:deps` | pip-audit over the packages installed in `.venv` |
| `security:secrets` | gitleaks over the whole git history, in its pinned image |

The gitleaks pre-commit hook covers staged changes at commit time; it cannot scan history.
The Rust dependencies are audited by `task rust:audit` (cargo-audit in the pinned image).
