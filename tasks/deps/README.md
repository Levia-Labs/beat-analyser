# tasks / deps

Namespace `deps:`. The host virtualenv `.venv`, which runs the CLI and the tests.
Tasks: `setup` creates it and installs app and dev dependencies (beatcore too when Rust is
present), `beatcore` builds and installs the Rust extension, `check` reports broken or
outdated packages without changing anything, `yt-dlp` upgrades yt-dlp when YouTube
downloads fail ([ADR 0004](../../docs/adr/0004-yt-dlp-unpinned.md)). See
[docs/tasks.md](../../docs/tasks.md).
