"""The optional web token: WEB_TOKEN set means the API needs a Bearer header or the session
cookie that /login hands out (docs/adr/0015)."""

import hashlib
import hmac
import os
import subprocess
import sys
from pathlib import Path

import netutil
import pytest
import webauth

import app as app_module

TOKEN = "s3cret-token"  # noqa: S105
APP_DIR = Path(app_module.__file__).resolve().parent
PROTECTED = [
    ("get", "/projects"),
    ("get", "/projects/x/events"),
    ("get", "/progress/abc"),
    ("get", "/folders"),
    ("get", "/download/x"),
    ("post", "/upload"),
    ("post", "/download_yt"),
    ("post", "/repair"),
    ("post", "/projects/x/rename"),
    ("delete", "/projects/x"),
]


@pytest.fixture
def secured(client, web_dist, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "WEB_TOKEN", TOKEN)
    monkeypatch.setattr(webauth, "WRONG_TOKEN_DELAY", 0)
    return client


def cookie_value(response):
    header = next(h for h in response.headers.getlist("Set-Cookie") if h.startswith("ba_session="))
    return header


def test_unset_token_leaves_everything_open(client):
    assert client.get("/projects").status_code == 200
    assert client.get("/config").get_json()["auth"] is False


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_protected_routes_refuse_without_a_token(secured, method, path):
    response = getattr(secured, method)(path)
    assert response.status_code == 401
    assert response.get_json() == {"error": "Sign in required"}


@pytest.mark.parametrize("path", ["/", "/add", "/p/x", "/assets/app-abc123.js", "/config"])
def test_open_routes_stay_open(secured, path):
    assert secured.get(path).status_code == 200


def test_config_reports_auth(secured):
    assert secured.get("/config").get_json()["auth"] is True


def test_bearer_header_is_accepted(secured):
    assert secured.get("/projects", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


@pytest.mark.parametrize("header", ["Bearer wrong", TOKEN, "Basic " + TOKEN, "Bearer "])
def test_wrong_bearer_is_refused(secured, header):
    assert secured.get("/projects", headers={"Authorization": header}).status_code == 401


def test_login_sets_a_hardened_session_cookie(secured):
    response = secured.post("/login", json={"token": TOKEN})
    assert response.status_code == 204
    cookie = cookie_value(response)
    expected = hmac.new(TOKEN.encode(), b"beat-analyser session", hashlib.sha256).hexdigest()
    assert cookie.startswith(f"ba_session={expected};")
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Path=/" in cookie
    assert "Max-Age" not in cookie
    assert "Expires" not in cookie
    assert "Secure" not in cookie
    assert TOKEN not in cookie
    assert secured.get("/projects").status_code == 200


def test_login_over_https_marks_the_cookie_secure(secured):
    response = secured.post("/login", json={"token": TOKEN}, base_url="https://localhost")
    assert "Secure" in cookie_value(response)


@pytest.mark.parametrize("body", [{"token": "wrong"}, {"token": ""}, {}, {"token": 5}, [1], None])
def test_wrong_login_is_refused_and_sets_no_cookie(secured, body):
    response = secured.post("/login", json=body)
    assert response.status_code == 401
    assert "Set-Cookie" not in response.headers
    assert secured.get("/projects").status_code == 401


def test_wrong_login_waits(secured, monkeypatch):
    waits = []
    monkeypatch.setattr(webauth, "WRONG_TOKEN_DELAY", 1)
    monkeypatch.setattr(webauth.time, "sleep", waits.append)
    secured.post("/login", json={"token": "wrong"})
    secured.post("/login", json={"token": TOKEN})
    assert waits == [1]


def test_default_delay_is_one_second():
    assert webauth.WRONG_TOKEN_DELAY == 1


def test_a_forged_cookie_is_refused(secured):
    secured.set_cookie("ba_session", "0" * 64)
    assert secured.get("/projects").status_code == 401
    secured.set_cookie("ba_session", TOKEN)
    assert secured.get("/projects").status_code == 401


def test_logout_clears_the_cookie(secured):
    secured.post("/login", json={"token": TOKEN})
    response = secured.post("/logout")
    assert response.status_code == 204
    assert "ba_session=;" in cookie_value(response)
    assert secured.get("/projects").status_code == 401


def test_login_and_logout_keep_the_same_origin_check(secured):
    headers = {"Origin": "http://evil.example"}
    assert secured.post("/login", json={"token": TOKEN}, headers=headers).status_code == 403
    assert secured.post("/logout", headers=headers).status_code == 403


def test_a_signed_in_write_still_needs_same_origin(secured):
    secured.post("/login", json={"token": TOKEN})
    response = secured.delete("/projects/x", headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_login_without_a_configured_token_is_refused(client):
    assert client.post("/login", json={"token": ""}).status_code == 401


def test_public_bind_without_a_token_is_refused():
    with pytest.raises(SystemExit, match="WEB_TOKEN"):
        webauth.check_bind("", "0.0.0.0")  # noqa: S104


@pytest.mark.parametrize(
    ("token", "bind"),
    [("", "127.0.0.1"), ("", "localhost"), (TOKEN, "0.0.0.0")],  # noqa: S104
)
def test_allowed_start_combinations(token, bind):
    webauth.check_bind(token, bind)


def test_importing_the_app_on_a_public_bind_without_a_token_exits():
    env = {**os.environ, "BIND": "0.0.0.0", "WEB_TOKEN": ""}  # noqa: S104
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=APP_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "WEB_TOKEN" in result.stderr


def test_is_loopback_lives_in_netutil():
    assert netutil.is_loopback("127.0.0.1")
    assert not netutil.is_loopback("0.0.0.0")  # noqa: S104
