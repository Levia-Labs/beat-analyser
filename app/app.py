import os
import zipfile
from flask import Flask, render_template, request, send_file
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
from madmom.features.notes import RNNPianoNoteProcessor, NotePeakPickingProcessor
import numpy as np
import csv

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
TRACKS_FOLDER = os.path.join(UPLOAD_FOLDER, "tracks")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRACKS_FOLDER, exist_ok=True)


def write_csv(filename, events):
    """Write a list of (time, label) tuples to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for time, label in events:
            writer.writerow([f"{time:.4f}", label])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    audio_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(audio_path)

    # Clear previous tracks
    for f_name in os.listdir(TRACKS_FOLDER):
        os.remove(os.path.join(TRACKS_FOLDER, f_name))

    # --- Percussion extraction ---
    onset_processor = CNNOnsetProcessor()
    peak_picking = OnsetPeakPickingProcessor(fps=100)
    activations = onset_processor(audio_path)
    times = peak_picking(activations)

    percussion_events = {"kick": [], "snare": [], "hh": []}
    for idx, t in enumerate(times):
        # Simple round-robin assignment as placeholder
        if idx % 3 == 0:
            percussion_events["kick"].append((t, "kick"))
        elif idx % 3 == 1:
            percussion_events["snare"].append((t, "snare"))
        else:
            percussion_events["hh"].append((t, "hh"))

    for track, events in percussion_events.items():
        write_csv(os.path.join(TRACKS_FOLDER, f"{track}.csv"), events)

    # --- Pitched instruments extraction ---
    piano_processor = RNNPianoNoteProcessor()
    piano_notes = piano_processor(audio_path)

    # Peak picking to discretize activations
    peak_picker = NotePeakPickingProcessor(threshold=0.5, min_distance=0.05)
    discrete_notes = peak_picker(piano_notes)

    instrument_tracks = {"bass": [], "chords": [], "melody": []}
    for note in discrete_notes:
        # Safe unpack: some tuples may have extra info
        onset_time = note[0]
        pitch = note[1] if len(note) > 1 else 60  # default to middle C
        label = f"note_{int(pitch)}"
        if pitch < 48:
            instrument_tracks["bass"].append((onset_time, label))
        elif pitch < 72:
            instrument_tracks["chords"].append((onset_time, label))
        else:
            instrument_tracks["melody"].append((onset_time, label))

    for track, events in instrument_tracks.items():
        write_csv(os.path.join(TRACKS_FOLDER, f"{track}.csv"), events)

    # --- Package all CSVs into a ZIP ---
    zip_path = os.path.join(UPLOAD_FOLDER, f"{os.path.splitext(file.filename)[0]}_tracks.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f_name in os.listdir(TRACKS_FOLDER):
            zipf.write(os.path.join(TRACKS_FOLDER, f_name), arcname=f_name)

    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)