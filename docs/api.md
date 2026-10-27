# API

Every error is JSON: `{"error": "<message>"}` with the status code below. Every request
that is not GET, HEAD or OPTIONS from another origin is refused with 403 (see
[security.md](security.md)). ZIP contents are described in [output.md](output.md).

The web app is the only client that sends `Accept: application/json` to the two analysis
endpoints. It gets `{"name": "<project>"}` instead of the ZIP and fetches the files later.
curl, the MCP jobs and `tasks/dev/e2e.py` send no such header and keep getting the ZIP.

## `POST /upload`

Multipart form, field `file`. Accepted extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.
Non-WAV files are decoded to WAV with ffmpeg. An optional `X-Progress-Id` header (32 lowercase
hex characters, chosen by the client) lets the client poll `GET /progress/<id>` while the
request is open.

| Status | When                                                                     |
| ------ | ------------------------------------------------------------------------ |
| 200    | ZIP attachment `<name>_tracks.zip`, or `{"name": "song"}` for JSON       |
| 400    | No file, unsupported extension, or a file name with no usable characters |
| 413    | Larger than `MAX_UPLOAD_MB`                                              |
| 429    | `MAX_ANALYSES` analyses already running; nothing changed, retry shortly  |
| 500    | Decoding or analysis failed; the message says why                        |

A name that already has a result is compared by the SHA-256 of the audio: the same audio
returns the cached ZIP, different audio under the same name replaces the project and is
analysed again ([ADR 0014](adr/0014-content-hash-projects.md)). Every request holds one of
the `MAX_ANALYSES` slots while its audio is placed and hashed, before any heavy work, so a
repeat request for a cached project can briefly be refused with 429 at the cap.

```bash
curl -F file=@song.mp3 -o song_tracks.zip http://localhost:8002/upload
```

## `POST /download_yt`

JSON body `{"url": "<YouTube URL>"}`. Accepted forms: `youtube.com/watch?v=`,
`youtu.be`, `youtube.com/shorts`, on `www.`, `m.` and `music.` hosts. The project is
named after the video ID. `Accept: application/json` and `X-Progress-Id` work as for
`/upload`.

| Status | When                                                                                        |
| ------ | ------------------------------------------------------------------------------------------- |
| 200    | ZIP attachment `<video id>_tracks.zip`, or `{"name": "<video id>"}` for JSON                |
| 400    | Not a YouTube URL, video longer than `YT_MAX_MINUTES`, or audio larger than `MAX_UPLOAD_MB` |
| 429    | `MAX_ANALYSES` analyses already running; nothing changed, retry shortly                     |
| 500    | Download or analysis failed                                                                 |

## `GET /config`

What the upload form needs to know:

```json
{ "extensions": [".flac", ".m4a", ".mp3", ".ogg", ".wav"], "max_upload_mb": 200, "auth": false }
```

`auth` is true when `WEB_TOKEN` is set. `/config` is open either way.

## Sign-in (only with `WEB_TOKEN`)

With `WEB_TOKEN` set, every route needs `Authorization: Bearer <token>` or the `ba_session`
cookie, except `/`, `/add`, `/p/<name>`, `/assets/*`, `GET /config`, `POST /login` and
`POST /logout`. Anything else without one answers 401 `{"error": "Sign in required"}`.
Without `WEB_TOKEN` there is no 401 and the two routes below answer 401 to every login.

`POST /login`, JSON `{"token": "..."}`: 204 and a `ba_session` session cookie (`HttpOnly`,
`SameSite=Strict`, path `/`, `Secure` over https, no expiry), or 401 `{"error": "Wrong token"}`.
Five failed credentials (any route) from one client address within 60 seconds lock it out:
429 `{"error": "Too many failed sign-ins; wait a minute"}` until the window passes.

`POST /logout`: 204, the cookie cleared.

```bash
curl -H "Authorization: Bearer $WEB_TOKEN" http://localhost:8002/projects
```

## `GET /projects`

One page of the projects, filtered and sorted, for the Projects table:

```json
{
  "items": [
    {
      "name": "song",
      "done": true,
      "key": "A minor",
      "bpm": 120,
      "duration": 215.4,
      "created": "2026-10-27T09:00:00Z",
      "size": 1048576
    }
  ],
  "total": 42
}
```

| Parameter            | Meaning                                                         |
| -------------------- | --------------------------------------------------------------- |
| `q`                  | Case-insensitive substring of the name                          |
| `key`                | An exact key (`A minor`) or a mode (`major`, `minor`)           |
| `status`             | `built` (ZIP exists) or `pending`                               |
| `bpm_min`, `bpm_max` | Inclusive bounds; a project without a BPM never matches a bound |
| `sort`               | `name`, `key`, `bpm`, `duration`, `created` (default), `size`   |
| `order`              | `asc` or `desc` (default)                                       |
| `offset`, `limit`    | Paging; `offset` >= 0, `limit` 1 to 200 (default 50)            |

`total` counts the filtered set, not the page. `duration` is seconds of audio and `null` for
projects built before it was stored; `created` is the mtime of `meta.json`, `null` without
one; `size` is the project folder plus its ZIP, in bytes. Unknown values sort last in either
order. A bad value answers `400` with `{"error": "..."}`.

## `GET /projects/<name>/events`

Every track's rows for the viewer: `[seconds, label ID]` pairs, and the table that decodes the
IDs. `meta` is `null` when the project has no `meta.json`.

```json
{
  "name": "song",
  "meta": { "name": "song", "key": "C", "bpm": 120 },
  "labels": { "0": "kick", "1": "snare" },
  "tracks": {
    "kick": [
      [0.52, 0],
      [1.02, 0]
    ],
    "snare": [[1.01, 1]]
  }
}
```

| Status | When                                                              |
| ------ | ----------------------------------------------------------------- |
| 200    | The body above                                                    |
| 404    | The name is invalid, or the project has no ZIP yet (not finished) |

## `GET /progress/<id>`

The state of the analysis whose request carried `X-Progress-Id: <id>`. Stages are
`decode`, `rhythm`, `notes`, `key`, `package`; a YouTube run starts with `download`.
`fraction` is the progress inside the current stage when known (the download), otherwise
`null`.

```json
{
  "stages": ["decode", "rhythm", "notes", "key", "package"],
  "stage": "notes",
  "done": 2,
  "total": 5,
  "fraction": null,
  "finished": false,
  "error": null,
  "elapsed": 12.4
}
```

| Status | When                                                                          |
| ------ | ----------------------------------------------------------------------------- |
| 200    | The state above; `finished` turns true with `error` set when the run failed   |
| 404    | Unknown or malformed ID, or finished more than 10 minutes ago (entry dropped) |

The registry lives in one process's memory ([ADR 0013](adr/0013-react-spa-served-by-flask.md)).

## `DELETE /projects/<name>`

Removes the project folder (audio, CSVs) and its ZIP.

| Status | When                                                          |
| ------ | ------------------------------------------------------------- |
| 204    | Deleted, no body                                              |
| 404    | The name is invalid or no such project                        |
| 409    | The project is being analysed; nothing was touched, try again |

## `POST /projects/<name>/rename`

JSON body `{"to": "<new name>"}`. The new name follows the project name rule: letters, digits,
dot, dash or underscore, up to 100 characters. Renaming to the current name changes nothing.
The folder moves, `meta.json` takes the new name, and the ZIP, if the project had one, is
rebuilt under it.

```json
{ "name": "new-name" }
```

| Status | When                                                           |
| ------ | -------------------------------------------------------------- |
| 200    | Renamed; the body carries the new name                         |
| 400    | The new name is missing or not allowed                         |
| 404    | The name is invalid or has no project folder                   |
| 409    | The project is being analysed, or the new name is already used |

A YouTube project is named after its video ID. After a rename the same URL no longer finds
it, so analysing that URL again starts afresh under the ID.

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
