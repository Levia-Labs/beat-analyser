![beat-analyser logo: a waveform turned into drum events](https://raw.githubusercontent.com/Levia-Labs/beat-analyser/main/docs/logo.svg)

# beat-analyser

**Turn a song into timed drum, beat and note events.**

[![validate](https://github.com/Levia-Labs/beat-analyser/actions/workflows/validate.yml/badge.svg)](https://github.com/Levia-Labs/beat-analyser/actions/workflows/validate.yml)
[![built on Crepidoma](https://img.shields.io/badge/built%20on-Crepidoma-6d28d9.svg)](https://github.com/Levia-Labs/Crepidoma)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/Levia-Labs/beat-analyser/blob/main/LICENSE)

Upload an audio file or paste a YouTube URL, get back a ZIP with one CSV per instrument
track: kick, snare, hi-hat, beat grid, bass, chords and melody, plus the detected tempo and
key.

Source, issues and full documentation: <https://github.com/Levia-Labs/beat-analyser>

## How it works

1. **Rhythm** comes from beatcore, a Rust extension built into the image: onset detection,
   per-band drum classification (kick, snare, hi-hat), tempo and beat tracking.
2. **Notes** come from Spotify's [basic-pitch](https://github.com/spotify/basic-pitch) model on
   onnxruntime, split by pitch into bass, chords and melody. The key is estimated from them.
3. **ffmpeg** decodes any input; **yt-dlp** fetches YouTube audio.
4. **Flask** under gunicorn serves a React web UI, with a viewer that draws each track on a timeline, and a JSON API on
   port 8000.

No GPU and no network access are needed for analysis. YouTube downloads need outbound
internet.

## Quick start

```bash
mkdir -p uploads
docker run -d --name beat-analyser \
  -p 127.0.0.1:8080:8000 \
  -v "$PWD/uploads:/app/uploads" \
  --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges:true \
  levialabs/beat-analyser:latest
```

Open <http://localhost:8080>, or call the API directly:

```bash
curl -F file=@song.mp3 -o song_tracks.zip http://localhost:8080/upload
```

The container runs as uid 1000. On Linux, the mounted `uploads/` directory must be writable
by that uid (`sudo chown 1000 uploads`).

A `docker-compose.yml` with the same hardening is in the source repository.

## Configuration

| Variable         | Default        | Meaning                                            |
| ---------------- | -------------- | -------------------------------------------------- |
| `MAX_UPLOAD_MB`  | `200`          | Largest upload, and largest YouTube audio download |
| `YT_MAX_MINUTES` | `20`           | Longest YouTube video accepted                     |
| `UPLOAD_ROOT`    | `/app/uploads` | Where projects are stored. Mount a volume here     |

## API

| Method | Path               | Does                                                                  |
| ------ | ------------------ | --------------------------------------------------------------------- |
| GET    | `/`                | Web UI                                                                |
| POST   | `/upload`          | Multipart field `file` (`.wav .mp3 .flac .ogg .m4a`). Returns the ZIP |
| POST   | `/download_yt`     | JSON `{"url": "..."}`. Returns the ZIP                                |
| GET    | `/folders`         | Every analysed project with file sizes and metadata                   |
| GET    | `/download/<name>` | A project's ZIP                                                       |
| POST   | `/repair`          | Reanalyse every stored project. Slow                                  |

Errors are JSON: `{"error": "..."}`. Results are cached by name, so a second upload of
`song.mp3` returns the first result immediately.

## Output

```text
song/
├── audio.wav     decoded source
├── meta.json     {"name": "song", "key": "A minor", "bpm": 120.02}
├── labels.json   {"0": "note_0", ..., "128": "kick", "129": "snare", "130": "hh", "131": "beat"}
├── kick.csv  snare.csv  hh.csv  beats.csv
└── bass.csv  chords.csv  melody.csv
```

Each CSV row is `<seconds>,<label ID>`, with no header. Times are rounded to 4 decimals with
trailing zeros dropped. `labels.json` decodes the IDs and is the same in every project: IDs 0
to 127 are notes by MIDI pitch, 128 to 131 are kick, snare, hi-hat and beat.

```csv
0.49,128
0.98,69
```

Bass holds pitches below 48, chords 48 to 71, melody 72 and up. Full schema:
[docs/output.md](https://github.com/Levia-Labs/beat-analyser/blob/main/docs/output.md).

## Security

The API has no authentication. Publish the port on `127.0.0.1` (as above) unless the
service sits behind something that authenticates. Cross-origin POSTs are refused, and the
container runs unprivileged and read-only apart from the uploads volume and `/tmp`.

## Tags

| Tag      | What it is                          |
| -------- | ----------------------------------- |
| `latest` | Current `main` build. `linux/amd64` |

## Limits

1. Drum classification is a three-band heuristic, not a drum transcriber. One onset gets one
   label, so a kick and hi-hat struck together count once.
2. One global tempo per song; tempo changes are not followed.
3. Drum-only audio gives key `unknown`.

## Part of Levia Labs

1. [Crepidoma](https://github.com/Levia-Labs/Crepidoma): the base every Levia Labs project
   stands on, a Claude Code plugin with shared Task namespaces, configs and ADRs.
2. [Levia-Labs on GitHub](https://github.com/Levia-Labs): every public project.

## Licence

MIT. The bundled basic-pitch model is Apache 2.0, Copyright Spotify AB.
