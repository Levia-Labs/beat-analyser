import os
import zipfile
from flask import Flask, render_template, request, send_file
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
from madmom.features.notes import RNNPianoNoteProcessor, NotePeakPickingProcessor
import csv

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

    base_name = os.path.splitext(file.filename)[0]
    file_folder = os.path.join(UPLOAD_FOLDER, base_name)
    os.makedirs(file_folder, exist_ok=True)

    # --- Percussion extraction ---
    onset_processor = CNNOnsetProcessor()
    peak_picking = OnsetPeakPickingProcessor(fps=100)
    activations = onset_processor(audio_path)
    times = peak_picking(activations)

    # Load audio for spectral analysis
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    def classify_percussion(t):
        """Classify percussive event based on energy in different frequency bands."""
        # Take a short window around onset (50ms)
        start = max(int((t - 0.025) * sr), 0)
        end = min(int((t + 0.025) * sr), len(y))
        frame = y[start:end]

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1/sr)

        # Compute energy in bands
        kick_energy = spectrum[(freqs >= 0) & (freqs <= 150)].sum()
        snare_energy = spectrum[(freqs > 150) & (freqs <= 4000)].sum()
        hh_energy = spectrum[(freqs > 4000)].sum()

        # Pick the max energy band
        energies = {"kick": kick_energy, "snare": snare_energy, "hh": hh_energy}
        return max(energies, key=energies.get)

    percussion_events = {"kick": [], "snare": [], "hh": []}
    for t in times:
        label = classify_percussion(t)
        percussion_events[label].append((t, label))

    for track, events in percussion_events.items():
        write_csv(os.path.join(file_folder, f"{track}.csv"), events)

    # --- Pitched instruments extraction ---
    piano_processor = RNNPianoNoteProcessor()
    piano_notes = piano_processor(audio_path)
    peak_picker = NotePeakPickingProcessor(threshold=0.5, min_distance=0.05)
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

    zip_path = os.path.join(UPLOAD_FOLDER, f"{base_name}_tracks.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(file_folder):
            for f_name in files:
                zipf.write(os.path.join(root, f_name), arcname=os.path.join(base_name, f_name))

    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)