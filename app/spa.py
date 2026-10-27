"""The built web app: the index for its client-side routes, hashed assets, and the security
headers every response carries. The routes' endpoint names (index, viewer, about, assets) are
the ones webauth.OPEN_ENDPOINTS leaves open so the sign-in page can load."""

from pathlib import Path

from flask import Flask, jsonify, send_file, send_from_directory

# Everything is same-origin: the SPA in web/ is built into hashed files Flask serves itself.
# No inline script or style, so none is allowed. React sets styles through the DOM, which a
# CSP governs only for style="" attributes in parsed HTML.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'self'"
)


def install(app: Flask) -> None:
    @app.after_request
    def security_headers(response):
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    def spa_index():
        index = Path(app.config["WEB_DIST"]) / "index.html"
        if not index.is_file():
            return jsonify(error="The web app is not built: run task web:build"), 503
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

    @app.get("/about")
    def about():
        # Static page: open without sign-in, like the shell it lives in.
        return spa_index()

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        # send_from_directory refuses paths that leave the folder (404).
        response = send_from_directory(Path(app.config["WEB_DIST"]) / "assets", filename)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
