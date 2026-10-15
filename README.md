# beat-analyser

Turns audio into timed rhythm and note events. Give it an MP3, WAV, FLAC, OGG or M4A file,
or a YouTube URL, and it returns a ZIP with one CSV per track: kick, snare, hi-hat, bass,
chords and melody.

Detection uses [CPJKU/madmom](https://github.com/CPJKU/madmom) neural networks; the web UI
is [Flask](https://flask.palletsprojects.com/). It is built for feeding game engines,
procedural music and transcription experiments with discrete events instead of raw audio.

Docker image: [levialabs/beat-analyser](https://hub.docker.com/r/levialabs/beat-analyser)

## Quick start

Needs [Docker](https://docs.docker.com/get-docker/) and [Task](https://taskfile.dev/installation/).

```bash
task docker:up        # first build compiles madmom and takes a while
```

Open <http://localhost>. Projects are saved under `uploads/` on the host.

Without Docker (Python 3.14 and ffmpeg on PATH):

```bash
task setup                     # .venv, dependencies, git hooks
task analyse FILE=song.mp3     # analyse from the command line
task run                       # web UI on http://127.0.0.1:5000
```

`task --list` shows everything else; [docs/tasks.md](docs/tasks.md) explains it.

## Output

Each analysed song becomes a folder and a ZIP:

```text
uploads/
├── song/
│   ├── audio.wav      the decoded source
│   ├── meta.json      {"name": "song", "key": "C", "bpm": 120} - key and bpm are placeholders
│   ├── kick.csv  snare.csv  hh.csv
│   └── bass.csv  chords.csv  melody.csv
└── song_tracks.zip    the folder above, zipped under song/
```

Every CSV row is `<time in seconds>,<label>`:

```csv
0.4900,kick
0.9800,note_69
```

1. Percussion labels: `kick`, `snare`, `hh`, chosen by which frequency band dominates 25 ms
   either side of the onset. Approximate by design.
2. Note labels: `note_<MIDI pitch>`. Bass is below 48, chords 48 to 71, melody 72 and up.

## API

| Method | Path | Does |
| --- | --- | --- |
| GET | `/` | Web UI |
| POST | `/upload` | Form field `file`. Returns the ZIP |
| POST | `/download_yt` | JSON `{"url": "..."}`. Returns the ZIP |
| GET | `/folders` | Every project with file sizes and `meta.json` |
| GET | `/download/<name>` | A project's ZIP |
| POST | `/repair` | Reanalyse every project. Slow |

Details and error codes: [docs/api.md](docs/api.md).

## Documentation

1. [docs/architecture.md](docs/architecture.md) - modules and the processing pipeline
2. [docs/configuration.md](docs/configuration.md) - environment variables
3. [docs/security.md](docs/security.md) - what is protected, what is not, before exposing it
4. [docs/troubleshooting.md](docs/troubleshooting.md)
5. [docs/development.md](docs/development.md) - working on the code
6. [docs/adr/](docs/adr/README.md) - why things are the way they are
7. [TODO.md](TODO.md) - known gaps

## Limits

1. Percussion classification is a three-band heuristic, not a drum transcriber.
2. Notes come from a piano transcription model; other instruments are approximated.
3. No tempo or key detection yet; `meta.json` carries placeholders.
4. A project is keyed by file name. Uploading a different file with an existing name
   returns the cached result; see [docs/troubleshooting.md](docs/troubleshooting.md).

## Licence

MIT, see [LICENSE](LICENSE).
