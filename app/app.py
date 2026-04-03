import os
import zipfile
import json
import numpy as np
import librosa
import csv

from flask import Flask, render_template, request, send_file
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
from madmom.features.notes import RNNPianoNoteProcessor, NotePeakPickingProcessor

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Utility functions ---

def write_csv(filename, events):
    """Write time-labeled events to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for time, label in events:
            writer.writerow([f"{time:.4f}", label])


def classify_percussion(t, sr, y):
    """Classify a percussive event around time t."""
    start = max(int((t - 0.025) * sr), 0)
    end = min(int((t + 0.025) * sr), len(y))
    frame = y[start:end]

    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1/sr)

    total_energy = spectrum.sum() + 1e-8
    kick_energy = spectrum[(freqs >= 0) & (freqs <= 150)].sum() / total_energy
    snare_energy = spectrum[(freqs > 150) & (freqs <= 4000)].sum() / total_energy
    hh_energy = spectrum[(freqs > 4000)].sum() / total_energy

    energies = {
        "kick": kick_energy * 1.2,
        "snare": snare_energy * 1.1,
        "hh": hh_energy
    }

    return max(energies, key=energies.get)


def zip_output(base_name, file_folder):
    """Compress folder contents into a zip file."""
    zip_path = os.path.join(UPLOAD_FOLDER, f"{base_name}_tracks.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(file_folder):
            for f_name in files:
                zipf.write(os.path.join(root, f_name), arcname=os.path.join(base_name, f_name))
    return zip_path


def save_meta(file_folder, base_name, key="C", bpm=120):
    """Create meta.json for a project."""
    meta = {
        "name": base_name,
        "key": key,
        "bpm": bpm
    }
    with open(os.path.join(file_folder, "meta.json"), "w") as f:
        json.dump(meta, f)


# --- Workflow functions ---

def extract_percussion(audio_path, file_folder):
    """Detect percussion events and save CSVs."""
    onset_processor = CNNOnsetProcessor()
    peak_picking = OnsetPeakPickingProcessor(fps=250, threshold=0.03, combine=1)
    activations = onset_processor(audio_path)
    times = peak_picking(activations)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    percussion_events = {"kick": [], "snare": [], "hh": []}

    for t in times:
        label = classify_percussion(t, sr, y)
        percussion_events[label].append((t, label))

    for track, events in percussion_events.items():
        write_csv(os.path.join(file_folder, f"{track}.csv"), events)


def extract_pitched_instruments(audio_path, file_folder):
    """Detect piano/notes events and save CSVs categorized by pitch."""
    piano_processor = RNNPianoNoteProcessor()
    piano_notes = piano_processor(audio_path)
    peak_picker = NotePeakPickingProcessor(threshold=0.15, min_distance=0.01)
    discrete_notes = peak_picker(piano_notes)

    instrument_tracks = {"bass": [], "chords": [], "melody": []}
    for note in discrete_notes:
        onset_time = note[0]
        pitch = note[1] if len(note) > 1 else 60
        label = f"note_{int(pitch)}"
        if pitch < 48:
            instrument_tracks["bass"].append((onset_time, label))
        elif pitch < 72:
            instrument_tracks["chords"].append((onset_time, label))
        else:
            instrument_tracks["melody"].append((onset_time, label))

    for track, events in instrument_tracks.items():
        write_csv(os.path.join(file_folder, f"{track}.csv"), events)


def process_upload(file):
    """Full workflow for handling an uploaded audio file."""
    base_name = os.path.splitext(file.filename)[0]
    file_folder = os.path.join(UPLOAD_FOLDER, base_name)
    os.makedirs(file_folder, exist_ok=True)

    audio_path = os.path.join(file_folder, "audio.wav")
    file.save(audio_path)

    save_meta(file_folder, base_name)

    extract_percussion(audio_path, file_folder)
    extract_pitched_instruments(audio_path, file_folder)

    zip_path = zip_output(base_name, file_folder)
    return zip_path


# --- Flask routes ---

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

    zip_path = process_upload(file)
    return send_file(zip_path, as_attachment=True)


@app.route("/folders", methods=["GET"])
def list_folders():
    """Return all project folders with file sizes and meta.json content."""
    result = {}
    for folder in os.listdir(UPLOAD_FOLDER):
        folder_path = os.path.join(UPLOAD_FOLDER, folder)
        if not os.path.isdir(folder_path):
            continue
        files_info = {}
        for f_name in os.listdir(folder_path):
            f_path = os.path.join(folder_path, f_name)
            if os.path.isfile(f_path):
                size = os.path.getsize(f_path)
                if f_name == "meta.json":
                    with open(f_path, "r") as f:
                        meta_content = json.load(f)
                    files_info[f_name] = {"size": size, "content": meta_content}
                else:
                    files_info[f_name] = {"size": size}
        result[folder] = files_info
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)