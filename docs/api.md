# API

Every error is JSON: `{"error": "<message>"}` with the status code below. Every POST from
another origin is refused with 403 (see [security.md](security.md)). ZIP contents are described
in [output.md](output.md).

## `POST /upload`

Multipart form, field `file`. Accepted extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.
Non-WAV files are decoded to WAV with ffmpeg.

| Status | When                                                                     |
| ------ | ------------------------------------------------------------------------ |
| 200    | ZIP attachment `<name>_tracks.zip`                                       |
| 400    | No file, unsupported extension, or a file name with no usable characters |
| 413    | Larger than `MAX_UPLOAD_MB`                                              |
| 500    | Decoding or analysis failed; the message says why                        |

```bash
curl -F file=@song.mp3 -o song_tracks.zip http://localhost/upload
```

## `POST /download_yt`

JSON body `{"url": "<YouTube URL>"}`. Accepted forms: `youtube.com/watch?v=`,
`youtu.be`, `youtube.com/shorts`, on `www.`, `m.` and `music.` hosts. The project is
named after the video ID.

| Status | When                                                                                        |
| ------ | ------------------------------------------------------------------------------------------- |
| 200    | ZIP attachment `<video id>_tracks.zip`                                                      |
| 400    | Not a YouTube URL, video longer than `YT_MAX_MINUTES`, or audio larger than `MAX_UPLOAD_MB` |
| 500    | Download or analysis failed                                                                 |

## `GET /folders`

Every project and its files:

```json
{
  "song": {
    "audio.wav": { "size": 52920044 },
    "kick.csv": { "size": 1180 },
    "meta.json": { "size": 44, "content": { "name": "song", "key": "C", "bpm": 120 } }
  }
}
```

## `GET /download/<name>`

The project's ZIP, or 404 when the name is invalid or no ZIP exists.

## `POST /repair`

Reanalyses every project folder that holds an `audio.wav`: CSVs deleted and regenerated, ZIP
rebuilt, `meta.json` kept (or created if missing). Runs synchronously; with many songs the
request takes a long time. `task repair` does the same without HTTP.

```json
{
  "repaired_projects": [
    { "folder": "song", "zip": "song_tracks.zip" },
    { "folder": "broken", "error": "..." }
  ]
}
```
