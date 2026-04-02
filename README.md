# beat-analyser

read music file beat data with python madmom package, dockerized

Web service that converts an uploaded audio file (MP3/WAV) into time-aligned drum events (kick, snare, hi-hat). Uses madmom for inference and Flask for the HTTP interface.

---

## Function

- Upload audio through browser  
- Extract percussive events via neural drum transcription  
- Return a text file with timestamps and labels  

Output is suitable for conversion into timeline markers (e.g., FMOD).

---

## Output Format

```csv

<time_seconds>,<label>

```

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
- Input: uploaded file saved locally  
- Output: generated text file returned to client  

---

## Project Structure

```bash

.
├── Dockerfile
├── app.py
└── templates/
└── index.html

```

---

## Build

```bash

docker build -t madmom-web .

```

---

## Run

```bash

docker run -p 5000:5000 madmom-web

```

Access: [http://localhost:5000](http://localhost:5000)

---

## API

### GET /

Returns upload page.

### POST /upload

Form-data:

- file: audio file (.mp3 or .wav)

Response:

- downloadable .txt file with drum events

---

## Processing Pipeline

1. Save uploaded file  
2. Load audio  
3. Run:
   - RNNDrumProcessor → activation map  
   - DBNDrumTrackingProcessor → discrete events  
4. Write events to file  
5. Return file  

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

- MP3 decoding depends on ffmpeg  
- Drum classification is probabilistic, not exact  
- Dense mixes reduce accuracy  
- No instrument separation beyond kick/snare/hat classes  
- No tempo normalization or quantization applied  

---

## Extension Points

- Add BPM detection (madmom beat processors)  
- Quantize timestamps to grid  
- Export JSON instead of plain text  
- Multi-file batch processing  
- Persistent storage instead of overwrite  
- Frontend visualization (timeline rendering)  

---

## Use Case

Transform raw audio into discrete rhythmic events for systems requiring explicit timing (game audio engines, procedural music systems).
