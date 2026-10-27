"""The optional web token (docs/adr/0015). With WEB_TOKEN set, the API needs either
`Authorization: Bearer <token>` or the `ba_session` cookie that POST /login sets.

The cookie, not a header alone, because the SPA's ZIP buttons are plain links, which cannot
carry an Authorization header. The cookie value is an HMAC of a fixed label keyed by the
token, so the token itself never sits in the browser and a stolen cookie is not the token.
"""

import hashlib
import hmac
import time

from flask import Flask, Response, jsonify, request
from netutil import is_loopback

COOKIE = "ba_session"
# The SPA shell must load to show the sign-in page; /config tells it whether to.
OPEN_ENDPOINTS = frozenset({"index", "add", "viewer", "assets", "client_config", "login", "logout"})
# Slows online guessing: each wrong token costs a second of the attacker's time.
WRONG_TOKEN_DELAY = 1


def check_bind(token: str, bind: str) -> None:
    """Refuse to serve the API to the network without a token (ADR 0008)."""
    if not token and not is_loopback(bind):
        raise SystemExit(
            f"BIND={bind} publishes the web app beyond this machine: "
            "set WEB_TOKEN (docs/security.md)"
        )


def session_value(token: str) -> str:
    return hmac.new(token.encode(), b"beat-analyser session", hashlib.sha256).hexdigest()


def same(given: str, expected: str) -> bool:
    # Constant time over bytes: compare_digest refuses non-ASCII str.
    return hmac.compare_digest(given.encode(), expected.encode())


def signed_in(token: str) -> bool:
    if same(request.headers.get("Authorization", ""), f"Bearer {token}"):
        return True
    return same(request.cookies.get(COOKIE, ""), session_value(token))


def install(app: Flask) -> None:
    """Register the gate and the login routes. The token is read per request from
    app.config["WEB_TOKEN"], so tests can switch it."""

    def error(message: str, status: int):
        return jsonify(error=message), status

    @app.before_request
    def require_token():
        token = app.config.get("WEB_TOKEN", "")
        if not token or request.method == "OPTIONS":
            return None
        # Unknown routes (endpoint None) fall through to the 404 handler.
        if request.endpoint is None or request.endpoint in OPEN_ENDPOINTS:
            return None
        if signed_in(token):
            return None
        return error("Sign in required", 401)

    @app.post("/login")
    def login():
        token = app.config.get("WEB_TOKEN", "")
        body = request.get_json(silent=True)
        given = body.get("token") if isinstance(body, dict) else None
        if not token or not isinstance(given, str) or not same(given, token):
            time.sleep(WRONG_TOKEN_DELAY)
            return error("Wrong token", 401)
        response = Response(status=204)
        response.set_cookie(
            COOKIE,
            session_value(token),
            httponly=True,
            samesite="Strict",
            path="/",
            secure=request.is_secure,
        )
        return response

    @app.post("/logout")
    def logout():
        response = Response(status=204)
        response.delete_cookie(COOKIE, path="/", httponly=True, samesite="Strict")
        return response
