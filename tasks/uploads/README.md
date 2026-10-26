# tasks / uploads

Namespace `uploads:`. Housekeeping for `uploads/`, the gitignored runtime data. `size`
shows disk use, `zips:clean` drops cached ZIPs (rebuilt from the CSVs on the next
request), `fixtures:clean` deletes projects named `fixture-*`.

`clean` deletes every analysed project, audio included. It needs `CONFIRM=yes` and is
denied to agents in [.claude/settings.json](../../.claude/settings.json); a person types it.
See [docs/tasks.md](../../docs/tasks.md).
