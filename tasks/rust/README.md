# tasks / rust

Namespace `rust:`. The beatcore crate in [beatcore/](../../beatcore/README.md), built in a
pinned Rust image so no host toolchain is needed. See [docs/tasks.md](../../docs/tasks.md).

| File           | Role                                                                                                                                                                                                           |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Taskfile.yml` | The `rust:` namespace: `test` (synthetic tempo, beat and onset checks), `check` (rustfmt and clippy), `fmt`, `lock`, `audit` (RustSec, needs network), `analyse FILE=...` (beatcore-cli on a WAV), `host:test` |
