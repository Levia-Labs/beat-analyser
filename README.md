<div align="center">

<img src="docs/logo.svg" alt="beat-analyser logo: a waveform turned into drum events" width="128">

# beat-analyser

**Turn a song into timed drum, beat and note events.**

[![validate](https://github.com/Levia-Labs/beat-analyser/actions/workflows/validate.yml/badge.svg)](https://github.com/Levia-Labs/beat-analyser/actions/workflows/validate.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/levialabs/beat-analyser?logo=docker&label=docker)](https://hub.docker.com/r/levialabs/beat-analyser)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?logo=python&logoColor=white)](docs/development.md)
[![Rust core](https://img.shields.io/badge/core-Rust-b7410e.svg?logo=rust&logoColor=white)](beatcore/README.md)
[![built on Crepidoma](https://img.shields.io/badge/built%20on-Crepidoma-6d28d9.svg)](https://github.com/Levia-Labs/Crepidoma)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

</div>

Give beat-analyser an MP3, WAV, FLAC, OGG or M4A file, or a YouTube URL. It returns a ZIP
with one CSV per track (kick, snare, hi-hat, beats, bass, chords, melody) plus the detected
tempo and key. Discrete events instead of raw audio, ready for game engines, procedural music
and transcription experiments.

## Highlights

1. **Rhythm in Rust.** beatcore, the project's own extension, finds onsets, drum classes,
   tempo and the beat grid.
2. **Notes from basic-pitch.** Spotify's [basic-pitch](https://github.com/spotify/basic-pitch)
   model runs on onnxruntime; the key is derived from the notes.
3. **One image, three ways in.** Web UI, HTTP API and CLI share the same pipeline.
4. **Plain output.** Every CSV row is `<seconds>,<label ID>`; `labels.json` decodes the IDs.
5. **Gated.** Lint, types, ast-grep rules, property tests and synthetic ground-truth checks
   run locally and in CI.

## Quick start

Needs [Docker](https://docs.docker.com/get-docker/) and [Task](https://taskfile.dev/installation/).

```bash
task docker:up        # first build compiles beatcore; later builds reuse it
```

Open <http://localhost:8002>. Each finished project opens in a viewer with one timeline lane
per track. Projects are saved under `uploads/` on the host. A prebuilt image is on
[Docker Hub](https://hub.docker.com/r/levialabs/beat-analyser).

On the host (Python 3.14 and ffmpeg on PATH; Rust optional):

```bash
task setup                     # .venv, dependencies, git hooks; beatcore if cargo exists
task analyse FILE=song.mp3     # analyse from the command line
task web:install && task web:build   # once: the React app in web/ (Node 22)
task run                       # web UI on http://127.0.0.1:5000 (needs beatcore built)
```

`task --list` shows everything else; [docs/tasks.md](docs/tasks.md) explains it.

## Output

```text
uploads/
├── song/
│   ├── audio.wav      the decoded source
│   ├── meta.json      {"name": "song", "key": "A minor", "bpm": 120.02}
│   ├── labels.json    {"0": "note_0", ..., "128": "kick", ...}
│   ├── kick.csv  snare.csv  hh.csv  beats.csv
│   └── bass.csv  chords.csv  melody.csv
└── song_tracks.zip    the folder above, zipped under song/
```

```csv
0.49,128
0.98,69
```

Percussion IDs are 128 to 130, beats 131, notes their MIDI pitch. Full schema:
[docs/output.md](docs/output.md).

## API

| Method | Path               | Does                                          |
| ------ | ------------------ | --------------------------------------------- |
| GET    | `/`                | Web UI                                        |
| POST   | `/upload`          | Form field `file`. Returns the ZIP            |
| POST   | `/download_yt`     | JSON `{"url": "..."}`. Returns the ZIP        |
| GET    | `/folders`         | Every project with file sizes and `meta.json` |
| GET    | `/download/<name>` | A project's ZIP                               |
| POST   | `/repair`          | Reanalyse every project. Slow                 |

Details and error codes: [docs/api.md](docs/api.md). Agents can connect over MCP:
[docs/mcp.md](docs/mcp.md). Read [docs/security.md](docs/security.md)
before exposing the service beyond loopback.

## Documentation

| Read                                               | For                                 |
| -------------------------------------------------- | ----------------------------------- |
| [docs/](docs/README.md)                            | Index of every document             |
| [docs/architecture.md](docs/architecture.md)       | Modules and the processing pipeline |
| [docs/configuration.md](docs/configuration.md)     | Environment variables               |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common failures                     |
| [docs/development.md](docs/development.md)         | Working on the code                 |
| [docs/adr/](docs/adr/README.md)                    | Why things are the way they are     |
| [TODO.md](TODO.md)                                 | Known gaps                          |

## Limits

1. Percussion is a three-band heuristic, not a drum transcriber: one onset gets one label.
2. One global tempo per song; tempo changes are not followed.
3. Drum-only audio has no notes, so its key is `unknown`.
4. Projects are keyed by file name; a new file with an existing name returns the cached
   result ([docs/troubleshooting.md](docs/troubleshooting.md)).

## Part of Levia Labs

| Project                                                                     | What it is                                                                                                                                              |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Crepidoma](https://github.com/Levia-Labs/Crepidoma)                        | The base every Levia Labs project stands on: a Claude Code plugin, shared Task namespaces, configs and ADRs. Pinned here as the `.crepidoma/` submodule |
| [levialabs/beat-analyser](https://hub.docker.com/r/levialabs/beat-analyser) | The prebuilt Docker image                                                                                                                               |
| [Levia-Labs on GitHub](https://github.com/Levia-Labs)                       | Every public project                                                                                                                                    |

Built with [basic-pitch](https://github.com/spotify/basic-pitch),
[Flask](https://flask.palletsprojects.com/), [yt-dlp](https://github.com/yt-dlp/yt-dlp),
[Task](https://taskfile.dev/) and [ast-grep](https://ast-grep.github.io/).

## Licence

MIT, see [LICENSE](LICENSE). The basic-pitch model in `app/models/` is Apache 2.0, Copyright
Spotify AB ([app/models/README.md](app/models/README.md)).
