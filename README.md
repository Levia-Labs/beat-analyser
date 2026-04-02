# beat-analyser

Python-based beat and note extraction service using madmom, Dockerized.

Converts MP3/WAV into time-aligned percussion and pitched instrument events. CSVs output per track. Folder per upload. ZIP package returned.

It utilises [CPJKU/madmom](https://github.com/CPJKU/madmom) for audio processing and [Flask](https://flask.palletsprojects.com/) for the web interface.

Madmom package is almost impossible to install in modern environment and this setup barely works.

---

## Function

* Browser upload
* Extract percussion (kick, snare, hi-hat)
* Extract pitched notes (bass, chords, melody)
* Output CSVs per track
* Package CSV folder into ZIP

---

## Output Format

```csv
<time_seconds>,<label>
```

Percussion example: `kick.csv`

```csv
0.5123,kick
1.1020,kick
```

Pitched notes example: `bass.csv`

```csv
0.7431,note_40
1.0020,note_36
```

Labels:

* Percussion: kick, snare, hh
* Pitched: bass (MIDI <48), chords (48–71), melody (≥72)

---

## Project Structure

```bash
.
├── app
│   ├── app.py
│   ├── Dockerfile
│   └── templates/index.html
├── uploads/
├── docker-compose.yml
└── README.md
```

* `uploads/`: synchronized folder for uploaded audio and generated CSVs/ZIPs

---

## Build

```bash
docker compose build
```

> WARNING! Initial docker build is very slow due to ffmpeg installation and takes ~30 minutes.
>
> Subsequent builds are faster due to layer caching.

---

## Run

```bash
docker compose up
```

Access UI: `http://localhost`

Files are saved in `uploads/` folder on a host. ZIP response is sent to browser.

---

## API

GET `/` – returns upload page.

POST `/upload` – form-data `file`: audio. Response: ZIP with folder named after uploaded file, containing CSVs for percussion and pitched tracks.

---

## Processing Pipeline

1. Save uploaded file in `uploads/`
2. Create folder named after file
3. Load audio
4. Percussion extraction:

   * `CNNOnsetProcessor` + `OnsetPeakPickingProcessor`
   * Assign kick/snare/hh
5. Pitched instruments:

   * `RNNPianoNoteProcessor` + `NotePeakPickingProcessor`
   * Assign bass/chords/melody
6. Write CSVs
7. Package folder into ZIP

---

## Dependencies

System:

* ffmpeg
* libsndfile1
* build-essential

Python:

* numpy
* scipy
* cython
* madmom
* flask
* librosa

---

## Constraints

* MP3 decoding requires ffmpeg
* Percussion classification is approximate
* Only kick, snare, hh, and MIDI notes
* No tempo quantization

---

## Extension Points

* Additional pitched instruments
* BPM detection and quantization
* Export JSON or other structured formats
* Multi-file batch processing
* Frontend timeline visualization

---

## Use Case

Raw audio → discrete rhythmic and melodic events for engines, procedural music, transcription analysis.
