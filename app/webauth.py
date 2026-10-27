"""The optional web token (docs/adr/0015). With WEB_TOKEN set, the API needs either
`Authorization: Bearer <token>` or the `ba_session` cookie that POST /login sets.

The cookie, not a header alone, because the SPA's ZIP buttons are plain links, which cannot
carry an Authorization header. The cookie value is an HMAC of a fixed label keyed by the
token, so the token itself never sits in the browser and a stolen cookie is not the token.
"""

import hashlib
import hmac
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from flask import Flask, Response, jsonify, request
from netutil import is_loopback

COOKIE = "ba_session"
# The SPA shell must load to show the sign-in page; /config tells it whether to.
OPEN_ENDPOINTS = frozenset({"index", "viewer", "assets", "client_config", "login", "logout"})
MIN_TOKEN_LENGTH = 16
LOCKED_MESSAGE = "Too many failed sign-ins; wait a minute"


class FailureLimiter:
    """Per-client failed-credential counter. Blocking a worker per guess would let
    guessing starve the service; this answers 429 at once. Bounded: expired entries are
    pruned and at most max_ips clients are tracked, the least recently failed dropped first."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        limit: int = 5,
        window: float = 60,
        max_ips: int = 1024,
    ) -> None:
        self.clock = clock
        self.limit = limit
        self.window = window
        self.max_ips = max_ips
        self.lock = threading.Lock()
        self.failures: OrderedDict[str, deque[float]] = OrderedDict()

    def prune(self, ip: str, now: float) -> None:
        stamps = self.failures.get(ip)
        while stamps and now - stamps[0] >= self.window:
            stamps.popleft()
        if stamps is not None and not stamps:
            del self.failures[ip]

    def blocked(self, ip: str) -> bool:
        with self.lock:
            self.prune(ip, self.clock())
            return len(self.failures.get(ip, ())) >= self.limit

    def fail(self, ip: str) -> None:
        with self.lock:
            now = self.clock()
            for known in list(self.failures):
                self.prune(known, now)
            self.failures.setdefault(ip, deque()).append(now)
            self.failures.move_to_end(ip)
            while len(self.failures) > self.max_ips:
                self.failures.popitem(last=False)

    def tracked(self) -> int:
        with self.lock:
            return len(self.failures)

    def ips(self) -> list[str]:
        with self.lock:
            return list(self.failures)


def check_bind(token: str, bind: str) -> None:
    """Refuse to serve the API to the network without a token (ADR 0008), and refuse a
    token too short to resist guessing."""
    if token and len(token) < MIN_TOKEN_LENGTH:
        raise SystemExit(f"WEB_TOKEN must be at least {MIN_TOKEN_LENGTH} characters")
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


def presented_credentials() -> bool:
    return bool(request.headers.get("Authorization") or request.cookies.get(COOKIE))


def install(app: Flask) -> None:
    """Register the gate and the login routes. The token is read per request from
    app.config["WEB_TOKEN"], so tests can switch it."""
    app.extensions["webauth_limiter"] = FailureLimiter()

    def error(message: str, status: int):
        return jsonify(error=message), status

    def limiter() -> FailureLimiter:
        return app.extensions["webauth_limiter"]

    # request.remote_addr only: forwarded-for headers are client-controlled, so trusting
    # them would let an attacker pick the bucket it is counted in.
    @app.before_request
    def require_token():
        token = app.config.get("WEB_TOKEN", "")
        if not token or request.method == "OPTIONS":
            return None
        # Unknown routes (endpoint None) fall through to the 404 handler.
        if request.endpoint is None or request.endpoint in OPEN_ENDPOINTS:
            return None
        ip = request.remote_addr or ""
        if limiter().blocked(ip):
            return error(LOCKED_MESSAGE, 429)
        if signed_in(token):
            return None
        if presented_credentials():
            limiter().fail(ip)
        return error("Sign in required", 401)

    @app.post("/login")
    def login():
        token = app.config.get("WEB_TOKEN", "")
        ip = request.remote_addr or ""
        if token and limiter().blocked(ip):
            return error(LOCKED_MESSAGE, 429)
        body = request.get_json(silent=True)
        given = body.get("token") if isinstance(body, dict) else None
        if not token or not isinstance(given, str) or not same(given, token):
            if token:
                limiter().fail(ip)
            return error("Wrong token", 401)
        response = Response(status=204)
        response.set_cookie(
            COOKIE,
            session_value(token),
            httponly=True,
            samesite="Strict",
            path="/",
            secure=bool(app.config.get("WEB_COOKIE_SECURE")) or request.is_secure,
        )
        return response

    @app.post("/logout")
    def logout():
        response = Response(status=204)
        response.delete_cookie(COOKIE, path="/", httponly=True, samesite="Strict")
        return response
