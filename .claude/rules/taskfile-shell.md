---
paths:
  - "Taskfile.yml"
  - "tasks/**/*.yml"
---

# Shell inside the Taskfiles

go-task runs embedded shell through its own interpreter (`mvdan.cc/sh`), and `{{...}}` is
substituted first.

- Not bash. No arrays, no `[[ ]]`, no `<<<`.
- Tasks run on Windows and Linux. Use `{{.PY}}` for the venv interpreter and `{{.SYS_PY}}`
  for the system one; never hard-code `.venv/bin/python` or `python3`.
- Included Taskfiles do not inherit root vars reliably: pass them in the `includes:` entry
  (`vars: { PY: "{{.PY}}" }`), as the root Taskfile does.
- **Top-level vars of every included Taskfile share one scope; the last one loaded wins.**
  Prefix them with the namespace (`RUST_RUN`, `EVAL_RUN`): an unprefixed `RUN` in
  `tasks/eval/` was silently replaced by `tasks/rust/`'s and ran the wrong image.
  `task --verbose --dry <name>` shows the rendered command.
- Commands run from the repository root, so paths are repo-relative.
- **`PATH` cannot be set in `env:`** - go-task ignores it. Resolve a pipx tool instead:

  ```yaml
  vars:
    RUFF:
      sh: command -v ruff 2>/dev/null || printf '%s' "${PIPX_BIN_DIR:-$HOME/.local/bin}/ruff"
  ```

  `tasks/lint/` and `tasks/precommit/` do this.
- A task that deletes data needs a `preconditions` check on `CONFIRM=yes` and a deny rule in
  `.claude/settings.json`.
