from flask import Flask, render_template, request, send_file
import os
from tempfile import NamedTemporaryFile
from madmom.features.beats import RNNDrumProcessor, DBNDrumTrackingProcessor
import numpy as np
from pydub import AudioSegment

app = Flask(__name__)
UPLOAD_FOLDER = "/app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hot reload of templates and static files enabled by default in debug mode
app.config["TEMPLATES_AUTO_RELOAD"] = True

# RNN + DBN processors (loaded once)
rnn_processor = RNNDrumProcessor()
dbn_processor = DBNDrumTrackingProcessor()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    # Save uploaded file to temp directory
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    # Convert non-wav files to wav for madmom
    if not temp_path.lower().endswith(".wav"):
        audio = AudioSegment.from_file(temp_path)
        wav_path = temp_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
    else:
        wav_path = temp_path

    # Run madmom drum transcription
    activations = rnn_processor(wav_path)
    events = dbn_processor(activations)

    # Map to kick/snare/hh based on activations index
    labels = ["kick", "snare", "hh"]
    output_file = NamedTemporaryFile(delete=False, suffix=".txt")
    with open(output_file.name, "w") as f:
        for t, idx in events:
            f.write(f"{t:.4f},{labels[idx]}\n")

    return send_file(output_file.name, as_attachment=True, download_name="drum_events.txt")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)