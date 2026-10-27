"""HTTP front end. Routes and request checks only: the work is in pipeline, analysis, storage
and youtube.

Configuration comes from the environment (docs/configuration.md). The server never runs with
the Werkzeug debugger: it was on, bound to 0.0.0.0, which is remote code execution for anyone
who can reach the port.
"""

import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

import analysis
import events
import pipeline
import storage
import youtube

MB = 1024 * 1024

# Bootstrap comes from jsDelivr with SRI; everything else is same-origin. No inline script or
# style, so none is allowed.
CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' https://cdn.jsdelivr.net; "
    "img-src 'self' data:; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("beat-analyser")

app = Flask(__name__)
app.config.update(
    UPLOAD_ROOT=Path(os.environ.get("UPLOAD_ROOT", "uploads")).resolve(),
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "200")) * MB,
    YT_MAX_SECONDS=int(os.environ.get("YT_MAX_MINUTES", "20")) * 60,
)


def upload_root() -> Path:
    root = Path(app.config["UPLOAD_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def error(message: str, status: int):
    return jsonify(error=message), status


def send_zip(path: Path):
    return send_file(path, as_attachment=True, download_name=path.name)


def wants_json() -> bool:
    """The SPA asks for JSON; curl, the MCP jobs and tasks/dev/e2e.py keep getting the ZIP."""
    return request.accept_mimetypes.best == "application/json"


def build_or_error(name: str, place_audio):
    try:
        path = pipeline.build(upload_root(), name, place_audio)
    except ValueError as exc:
        return error(str(exc), 400)
    except Exception as exc:
        log.exception("processing %s failed", name)
        return error(f"Processing failed: {exc}", 500)
    return jsonify(name=name) if wants_json() else send_zip(path)


@app.before_request
def same_origin_posts_only():
    """Refuse cross-site POSTs. A page on any other site could otherwise submit the upload
    form or trigger /repair against a server on localhost."""
    if request.method != "POST":
        return None
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return error("Cross-origin request refused", 403)
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return error("Cross-origin request refused", 403)
    return None


@app.after_request
def security_headers(response):
    response.headers.setdefault("Content-Security-Policy", CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.errorhandler(HTTPException)
def http_error(exc: HTTPException):
    return error(exc.description or exc.name, exc.code or 500)


@app.get("/")
def index():
    return render_template("index.html", accept=",".join(sorted(storage.ALLOWED_EXTENSIONS)))


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return error("No file uploaded", 400)
    extension = Path(file.filename).suffix.lower()
    if extension not in storage.ALLOWED_EXTENSIONS:
        return error(
            f"Unsupported file type; use {', '.join(sorted(storage.ALLOWED_EXTENSIONS))}", 400
        )
    try:
        name = storage.project_name(file.filename)
    except ValueError as exc:
        return error(str(exc), 400)

    def place_audio(wav: Path) -> None:
        if extension == ".wav":
            file.save(wav)
            return
        source = wav.with_name(f"source{extension}")
        file.save(source)
        try:
            analysis.to_wav(source, wav)
        finally:
            source.unlink(missing_ok=True)

    return build_or_error(name, place_audio)


@app.post("/download_yt")
def download_yt():
    data = request.get_json(silent=True) or {}
    try:
        vid = youtube.video_id(str(data.get("url", "")))
    except ValueError as exc:
        return error(str(exc), 400)

    def place_audio(wav: Path) -> None:
        source = youtube.download_audio(
            vid, wav.parent, app.config["YT_MAX_SECONDS"], app.config["MAX_CONTENT_LENGTH"]
        )
        try:
            analysis.to_wav(source, wav)
        finally:
            source.unlink(missing_ok=True)

    return build_or_error(vid, place_audio)


@app.get("/folders")
def list_folders():
    return jsonify(storage.list_projects(upload_root()))


@app.get("/config")
def client_config():
    return jsonify(
        extensions=sorted(storage.ALLOWED_EXTENSIONS),
        max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // MB,
    )


@app.get("/projects")
def list_projects():
    return jsonify(events.project_list(upload_root()))


@app.get("/projects/<name>/events")
def project_events(name: str):
    root = upload_root()
    # Only the ZIP means done: pipeline.build writes the CSVs before it packages.
    if not storage.is_valid_name(name) or not storage.zip_path(root, name).is_file():
        return error("Project not found", 404)
    return jsonify(events.project_events(root, name))


@app.get("/download/<name>")
def download_zip(name: str):
    path = storage.zip_path(upload_root(), name)
    if not storage.is_valid_name(name) or not path.is_file():
        return error("ZIP file not found", 404)
    return send_zip(path)


@app.post("/repair")
def repair_all_projects():
    return jsonify(repaired_projects=pipeline.repair_all(upload_root()))


if __name__ == "__main__":
    # Development server. The container runs gunicorn (Dockerfile).
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))
