# Output files

What one analysed song produces. Written by `analysis.write_tracks` and `storage.save_meta`,
packaged by `storage.package`.

## Layout

A project is a folder under the upload root plus a sibling ZIP. `<name>` is the sanitised
upload file stem, or the video ID for a YouTube URL.

```text
uploads/
├── <name>/
│   ├── audio.wav
│   ├── meta.json
│   ├── labels.json
│   ├── kick.csv  snare.csv  hh.csv  beats.csv
│   └── bass.csv  chords.csv  melody.csv
└── <name>_tracks.zip
```

The ZIP holds every file of the folder under a `<name>/` prefix. It is written to
`<name>_tracks.zip.part` and renamed, so a reader never sees a partial archive. All seven CSVs
are always written, empty when no event was found.

| File | Content |
| --- | --- |
| `audio.wav` | The source decoded to PCM WAV by ffmpeg; the input to analysis and `/repair` |
| `meta.json` | Song-level results, below |
| `labels.json` | Label ID to label name, decodes the CSVs, below |
| `*.csv` | One event track each, below |

## `meta.json`

```json
{"name": "song", "key": "A minor", "bpm": 120.02}
```

| Field | Type | Value |
| --- | --- | --- |
| `name` | string | The project name, same as the folder |
| `key` | string | `"<tonic> major"` or `"<tonic> minor"`, tonic one of `C C# D D# E F F# G G# A A# B`; `"unknown"` when no profile correlates well enough (drum-only audio) |
| `bpm` | number | beatcore's global tempo, 2 decimals; `0.0` when not analysed |

beatcore also returns a tempo confidence; it is not stored.

## `labels.json`

One object, label ID (as a JSON string key) to label name. The table is fixed in
`analysis.LABELS`, identical in every project, so one ID means the same in every song.

```json
{"0": "note_0", "1": "note_1", "...": "...", "127": "note_127",
 "128": "kick", "129": "snare", "130": "hh", "131": "beat"}
```

| ID | Label |
| --- | --- |
| 0 to 127 | `note_<ID>`: the ID is the MIDI pitch (60 is middle C, 69 is A4) |
| 128 | `kick` |
| 129 | `snare` |
| 130 | `hh` |
| 131 | `beat` |

## Track CSVs

| Property | Value |
| --- | --- |
| Header | None |
| Columns | `time,label_id` |
| `time` | Seconds from the start of the audio, rounded to 4 decimals, trailing zeros dropped (`0.49`, `2`, `0.5123`) |
| `label_id` | Integer, a key of `labels.json` |
| Encoding | UTF-8, `\r\n` line endings (Python `csv` default) |

Each row is one instant. There is no duration, end time, velocity or amplitude column.

```csv
0.49,128
0.98,69
```

| File | Label IDs | Source |
| --- | --- | --- |
| `kick.csv` | 128 | beatcore onsets classified as kick |
| `snare.csv` | 129 | beatcore onsets classified as snare |
| `hh.csv` | 130 | beatcore onsets classified as hi-hat |
| `beats.csv` | 131 | beatcore beat grid, one row per beat |
| `bass.csv` | 0 to 47 | basic-pitch notes, pitch below 48 |
| `chords.csv` | 48 to 71 | pitch 48 to 71 |
| `melody.csv` | 72 to 127 | pitch 72 and up |

Projects analysed before label IDs were introduced hold label names in the second column and
no `labels.json`. `/repair` (`task repair`) rewrites them in the current format.

1. Each percussion onset gets exactly one label, so simultaneous hits appear in one file only.
2. Note rows are sorted by start time, then pitch. A chord is several rows sharing a time.
3. Note end time and amplitude are computed by `notes.transcribe` and dropped on write.
