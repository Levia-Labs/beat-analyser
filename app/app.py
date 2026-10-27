"""HTTP front end. Routes and request checks only: the work is in pipeline, analysis, storage
and youtube.

Configuration comes from the environment (docs/configuration.md). The server never runs with
the Werkzeug debugger: it was on, bound to 0.0.0.0, which is remote code execution for anyone
who can reach the port.
"""

import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

import spa
import webauth
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

import analysis
import catalog
import events
import pipeline
import progress
import storage
import youtube

MB = 1024 * 1024

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
    WEB_TOKEN=os.environ.get("WEB_TOKEN", ""),
    WEB_COOKIE_SECURE=os.environ.get("WEB_COOKIE_SECURE") == "1",
)
webauth.check_bind(app.config["WEB_TOKEN"], os.environ.get("BIND", "127.0.0.1"))

# Analyses are CPU-bound and hold a decoded track in memory; a burst of uploads would starve
# the host. Taken without waiting, so an excess request answers 429 instead of queueing.
ANALYSES = threading.BoundedSemaphore(max(1, int(os.environ.get("MAX_ANALYSES", "2"))))
BUSY_MESSAGE = "Too many analyses running; try again shortly"


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
        path = pipeline.build(upload_root(), name, place_audio, tracker.stage, ANALYSES)
    except pipeline.TooBusy:
        tracker.finish(BUSY_MESSAGE)
        return error(BUSY_MESSAGE, 429)
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
    # same-site (a sibling subdomain) is refused too: only this origin or a user action.
    if request.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
        return error("Cross-origin request refused", 403)
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return error("Cross-origin request refused", 403)
    return None


# After the origin check: a cross-site write is refused before the token is looked at.
webauth.install(app)


spa.install(app)


@app.errorhandler(HTTPException)
def http_error(exc: HTTPException):
    return error(exc.description or exc.name, exc.code or 500)


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
        auth=bool(app.config["WEB_TOKEN"]),
    )


@app.get("/projects")
def list_projects():
    try:
        return jsonify(catalog.page(upload_root(), catalog.parse(request.args)))
    except catalog.QueryError as exc:
        return error(str(exc), 400)


@app.get("/projects/<name>/events")
def project_events(name: str):
    root = upload_root()
    # Only the ZIP means done: pipeline.build writes the CSVs before it packages.
    if not storage.is_valid_name(name) or not storage.zip_path(root, name).is_file():
        return error("Project not found", 404)
    return jsonify({**events.project_events(root, name), **catalog.facts(root, name)})


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
    body = request.get_json(silent=True)
    wanted = body.get("to") if isinstance(body, dict) else None
    target = wanted.strip() if isinstance(wanted, str) else ""
    # Renaming moves the folder, so a ZIP-only project has nothing to move.
    if not project_exists(root, name) or not storage.project_dir(root, name).is_dir():
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
