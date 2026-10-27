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

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

import analysis
import events
import pipeline
import progress
import storage
import youtube

MB = 1024 * 1024

# Everything is same-origin: the SPA in web/ is built into hashed files Flask serves itself.
# No inline script or style, so none is allowed. React sets styles through the DOM, which a
# CSP governs only for style="" attributes in parsed HTML.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'self'"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("beat-analyser")

app = Flask(__name__, static_folder=None)
app.config.update(
    WEB_DIST=Path(
        os.environ.get("WEB_DIST", Path(__file__).resolve().parent.parent / "web" / "dist")
    ),
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


def build_or_error(name: str, place_audio, tracker: progress.Tracker):
    try:
        path = pipeline.build(upload_root(), name, place_audio, tracker.stage)
    except ValueError as exc:
        tracker.finish(str(exc))
        return error(str(exc), 400)
    except Exception as exc:
        log.exception("processing %s failed", name)
        tracker.finish(f"Processing failed: {exc}")
        return error(f"Processing failed: {exc}", 500)
    tracker.finish()
    return jsonify(name=name) if wants_json() else send_zip(path)


@app.before_request
def same_origin_posts_only():
    """Refuse cross-site writes. A page on any other site could otherwise submit the upload
    form, trigger /repair or DELETE a project on a server on localhost."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
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


def spa_index():
    index = Path(app.config["WEB_DIST"]) / "index.html"
    if not index.is_file():
        return error("The web app is not built: run task web:build", 503)
    response = send_file(index, mimetype="text/html", max_age=0)
    # Asset names carry a hash; the index is what must never be stale.
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return spa_index()


@app.get("/p/<name>")
def viewer(name: str):
    # The SPA asks /projects/<name>/events, which validates the name.
    return spa_index()


@app.get("/assets/<path:filename>")
def assets(filename: str):
    # send_from_directory refuses paths that leave the folder (404).
    response = send_from_directory(Path(app.config["WEB_DIST"]) / "assets", filename)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


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

    tracker = progress.Tracker(request.headers.get("X-Progress-Id"), progress.UPLOAD_STAGES)

    def place_audio(wav: Path) -> None:
        tracker.stage("decode")
        if extension == ".wav":
            file.save(wav)
            return
        source = wav.with_name(f"source{extension}")
        file.save(source)
        try:
            analysis.to_wav(source, wav)
        finally:
            source.unlink(missing_ok=True)

    return build_or_error(name, place_audio, tracker)


@app.post("/download_yt")
def download_yt():
    data = request.get_json(silent=True) or {}
    try:
        vid = youtube.video_id(str(data.get("url", "")))
    except ValueError as exc:
        return error(str(exc), 400)

    tracker = progress.Tracker(request.headers.get("X-Progress-Id"), progress.YOUTUBE_STAGES)

    def place_audio(wav: Path) -> None:
        tracker.stage("download")
        source = youtube.download_audio(
            vid,
            wav.parent,
            app.config["YT_MAX_SECONDS"],
            app.config["MAX_CONTENT_LENGTH"],
            on_progress=tracker.fraction,
        )
        tracker.stage("decode")
        try:
            analysis.to_wav(source, wav)
        finally:
            source.unlink(missing_ok=True)

    return build_or_error(vid, place_audio, tracker)


@app.get("/progress/<progress_id>")
def analysis_progress(progress_id: str):
    state = progress.snapshot(progress_id)
    return jsonify(state) if state else error("No such progress", 404)


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


def project_exists(root: Path, name: str) -> bool:
    return storage.is_valid_name(name) and (
        storage.project_dir(root, name).is_dir() or storage.zip_path(root, name).is_file()
    )


@app.delete("/projects/<name>")
def delete_project(name: str):
    root = upload_root()
    if not project_exists(root, name):
        return error("Project not found", 404)
    try:
        pipeline.delete(root, name)
    except pipeline.Busy:
        return error("Project is being analysed", 409)
    return "", 204


@app.post("/projects/<name>/rename")
def rename_project(name: str):
    root = upload_root()
    target = str((request.get_json(silent=True) or {}).get("to", "")).strip()
    if not project_exists(root, name):
        return error("Project not found", 404)
    if not storage.is_valid_name(target):
        return error("Use letters, digits, dot, dash or underscore, up to 100 characters", 400)
    if target == name:
        return jsonify(name=name)
    try:
        pipeline.rename(root, name, target)
    except pipeline.Busy:
        return error("Project is being analysed", 409)
    except FileExistsError:
        return error(f"A project called {target} exists", 409)
    return jsonify(name=target)


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
