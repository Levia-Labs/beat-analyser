# Troubleshooting

First: `task doctor` lists what is installed and what is missing, and changes nothing.

## Analysis

### A new upload returns an old result

Projects are keyed by file name, and an existing ZIP is served as is
([ADR 0005](adr/0005-name-keyed-projects.md)). Rename the file, or delete
`uploads/<name>_tracks.zip` and `uploads/<name>/audio.wav` and upload again.

### `RuntimeError: beatcore is not installed`

The Rust extension is not built into this Python. In the container it always is. On the
host: install a Rust toolchain, then `task deps:beatcore`; without one, `task analyse`,
`task analyse:yt` and `task repair` run in the image (`task docker:build` first).
`import beatcore` succeeding is not proof: from the repository root, the crate's source
directory imports as an empty namespace package. `task doctor` checks properly.

### The tempo is half or double the real one

The tempo prior is centred on 120 BPM, one octave wide; a song at 70 BPM with busy
eighth notes can come out at 140. beats.csv then has twice the beats. Tune
`PRIOR_BPM` in `beatcore/src/tempo.rs` against real music before changing it - a
synthetic test passes either way.

### Beats fall between the kicks

The beat envelope weights kick, snare and hi-hat bands 1.0 / 0.7 / 0.35
(`BEAT_WEIGHTS` in `beatcore/src/features.rs`). Unweighted, off-beat hi-hats put every
beat half a beat late; `task rust:test` catches that on a synthetic loop.

### Hi-hats come out as snares (or the reverse)

Classification compares rise per frequency bin between bands (`DENSITY_WEIGHTS` in
`beatcore/src/onset.rs`, band edges in `beatcore/src/features.rs`). Change them only
against `task eval:synthetic` and `task docker:e2e` together: the eval clips are WAV and
the e2e fixture is MP3, and an MP3's empty top octave once turned every hat into a snare.

### Results look wrong after changing analysis code

ZIPs are a cache. `task repair` regenerates every project; for one project delete its ZIP
and run `task analyse` on its `audio.wav`.

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

### First build takes a long time

The first build pulls the Rust image, installs ffmpeg and compiles beatcore with
link-time optimisation. Later builds reuse every layer except the ones whose inputs
changed: an edit under `beatcore/` recompiles only the crate.

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
