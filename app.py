import os
from flask import Flask, request, render_template, send_file
from madmom.features.drums import RNNDrumProcessor, DBNDrumTrackingProcessor

app = Flask(__name__)
UPLOAD_PATH = "input.wav"
OUTPUT_PATH = "beats.txt"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    file.save(UPLOAD_PATH)

    act = RNNDrumProcessor()(UPLOAD_PATH)
    drums = DBNDrumTrackingProcessor()(act)

    with open(OUTPUT_PATH, "w") as f:
        for t, d in drums:
            f.write(f"{t:.4f},{d}\n")

    return send_file(OUTPUT_PATH, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
