# beat-analyser

Python-based beat extraction service using madmom, Dockerized.

Web service converts uploaded audio files (MP3/WAV) into time-aligned drum events (kick, snare, hi-hat). Uses madmom for drum transcription and Flask as the HTTP interface.

---

## Function

- Upload audio through browser  
- Extract percussive events via neural drum transcription  
- Return a text file with timestamps and drum labels  

Output is compatible with timeline-based systems (e.g., FMOD).

---

## Output Format

```csv
<time_seconds>,<label>
````

Example:

```csv
0.5123,kick
0.7431,hh
1.0020,snare
```

Labels:

- kick
- snare
- hh (hi-hat)

---

## Architecture

- HTTP server: Flask
- Processing: madmom (RNN + DBN drum tracking)
- Runtime: Docker container
- Orchestration: optional Docker Compose
- Input: uploaded audio saved inside container
- Output: generated text file returned to client

---

## Project Structure

```bash
.
├── Dockerfile
├── docker-compose.yml
├── app.py
└── templates/
    └── index.html
```

---

## Docker Compose (optional)

```yaml
version: "3.9"
services:
  madmom-web:
    build: .
    container_name: madmom-web
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app
    restart: unless-stopped
```

---

## Build

```bash
docker build -t madmom-web .
```

Or with Compose:

```bash
docker compose build
```

---

## Run

```bash
docker run -p 5000:5000 madmom-web
```

Or with Compose:

```bash
docker compose up
```

Access via: [http://localhost:5000](http://localhost:5000)

---

## API

### GET /

Returns the upload page.

### POST /upload

Form-data:

- `file`: audio file (.mp3 or .wav)

Response:

- downloadable `.txt` file containing drum events

---

## Processing Pipeline

1. Save uploaded file
2. Load audio
3. Run:

   - `RNNDrumProcessor` → activation map
   - `DBNDrumTrackingProcessor` → discrete drum events
4. Write events to file (`<timestamp>,<label>`)
5. Return file to client

---

## Dependencies

System:

- ffmpeg
- libsndfile1
- build-essential

Python:

- numpy
- scipy
- cython
- madmom
- flask

---

## Constraints

- MP3 decoding relies on ffmpeg
- Drum classification is probabilistic, not exact
- Accuracy drops on dense mixes
- Only kick, snare, hi-hat classification
- No tempo normalization or quantization applied

---

## Extension Points

- BPM detection and quantization
- Export JSON or other structured formats
- Multi-file batch processing
- Persistent storage for input/output
- Frontend timeline visualization

---

## Use Case

Convert raw audio into discrete rhythmic events for systems requiring explicit timing, e.g., game audio engines or procedural music systems.
