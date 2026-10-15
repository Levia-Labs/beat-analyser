# Troubleshooting

First: `task doctor` lists what is installed and what is missing, and changes nothing.

## Analysis

### A new upload returns an old result

Projects are keyed by file name, and an existing ZIP is served as is
([ADR 0005](adr/0005-name-keyed-projects.md)). Rename the file, or delete
`uploads/<name>_tracks.zip` and `uploads/<name>/audio.wav` and upload again.

### Every onset is late, early or missing

Check that the peak pickers in `app/analysis.py` use `ACTIVATION_FPS = 100`. Both madmom
networks emit 100 frames per second; a different `fps` scales every time, and a large
`combine` merges onsets. The previous code used `fps=250, combine=1`, which on a
four-beat test returned one onset at 0.004 s.

### Results look wrong after changing analysis code

ZIPs are a cache. `task repair` regenerates every project; for one project delete its ZIP
and run `task analyse` on its `audio.wav`.

### `VisibleDeprecationWarning: dtype(): align should be passed as ...`

madmom unpickling its models under NumPy 2.4. Harmless. The tasks pass `-W ignore`; RTK's
filter in [.rtk/filters.toml](../.rtk/filters.toml) drops it.

### `FileNotFoundError: ... 'ffmpeg'`

ffmpeg is not on PATH. Needed for every format except WAV and for YouTube. The container
has it; on the host install it and reopen the shell.

## YouTube

### Downloads fail with an extractor error

YouTube changed and the installed yt-dlp is behind. `task deps:yt-dlp` on the host;
`task docker:rebuild` for the container. Recent yt-dlp releases may also want a JavaScript
runtime (deno) for some videos; see the yt-dlp release notes.

### `Video longer than 20 minutes`

Raise `YT_MAX_MINUTES` ([configuration.md](configuration.md)).

## Container

### First build takes a very long time

madmom compiles from a pinned git commit. Later builds reuse that layer unless
`app/requirements.txt` changes.

### `PermissionError` writing to `/app/uploads`

The container runs as uid 1000. On a Linux host, `uploads/` must be writable by uid 1000:
`sudo chown -R 1000:1000 uploads`. Docker Desktop on Windows and macOS is not affected.

### Port 80 is already in use

`PORT=8080 task docker:up`, then <http://localhost:8080>.

### The web UI cannot be reached from another machine

By design: the port is published on `127.0.0.1`. Read [security.md](security.md), then
`BIND=0.0.0.0 task docker:up`.

## Development

### `task test` fails with `No module named pytest`

`task setup` (or `task deps:setup`) has not run in this clone.

### The Stop hook blocked the turn

[tasks/claude/hooks.py](../tasks/claude/hooks.py) ran `task validate` after an edit and it
failed; the output is attached. Run `task validate` and fix what it names. A missing tool is
reported, not blocked.

### `task validate` fails on a documentation finding

A document names a path that does not exist (case-exact), or a script is missing from its
folder's README. `task quality:docs` lists findings.
