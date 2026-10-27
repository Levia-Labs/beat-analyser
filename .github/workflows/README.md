# .github / workflows

| File           | What it does                                                                      |
| -------------- | --------------------------------------------------------------------------------- |
| `validate.yml` | Push to `main`, pull requests, manual dispatch. Jobs `gates`, `rust` and `image`. |

`gates` runs the same gates as `task validate` (lint, types, ast-grep rules and their tests, pytest, structure gates,
compose config). `rust` runs `cargo fmt --check`, clippy with `-D warnings` and `cargo test` in `beatcore/`. `image` builds
`app/Dockerfile` with buildx and a GitHub Actions layer cache (`type=gha`, `mode=max`), without pushing or loading. Actions
are pinned by commit.

`../dependabot.yml` opens weekly updates for pip (`/app`, `/`), cargo, npm, docker and github-actions, minor and patch
grouped, with yt-dlp ignored ([ADR 0004](../../docs/adr/0004-yt-dlp-unpinned.md)).
