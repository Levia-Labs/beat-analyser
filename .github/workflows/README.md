# .github / workflows

`validate.yml` runs on push to `main`, on pull requests and on manual dispatch. The `gates` job runs the same gates as
`task validate` (lint, types, ast-grep rules and their tests, pytest, structure gates, compose config). The `rust` job runs
`cargo fmt --check`, clippy with `-D warnings` and `cargo test` in `beatcore/`. The `image` job builds the Docker image
on manual dispatch only. Actions are pinned by commit.
