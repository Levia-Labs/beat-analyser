# .rtk

Holds `filters.toml`, project-local RTK output filters. They strip only success noise from gate output, so a failing
line is always shown. RTK applies them only after a person runs `rtk trust` on each machine; the trust is pinned to the
file hash, so run it again after every edit. See "Finding things cheaply" in [CLAUDE.md](../CLAUDE.md).
