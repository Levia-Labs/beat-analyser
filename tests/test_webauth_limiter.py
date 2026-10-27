"""The failed-sign-in limiter of the web token (docs/adr/0015): 5 failures per client IP per
minute, then 429, without sleeping."""

from pathlib import Path

import pytest
import webauth

import app as app_module

TOKEN = "s3cret-token-0123456"  # noqa: S105
BEARER = {"Authorization": f"Bearer {TOKEN}"}
LOCKED = {"error": "Too many failed sign-ins; wait a minute"}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def secured(client, web_dist, monkeypatch, clock):
    monkeypatch.setitem(app_module.app.config, "WEB_TOKEN", TOKEN)
    monkeypatch.setitem(app_module.app.extensions, "webauth_limiter", webauth.FailureLimiter(clock))
    return client


def fail_n(client, n, **kwargs):
    for _ in range(n):
        client.get("/projects", headers={"Authorization": "Bearer wrong"}, **kwargs)


def test_nothing_in_webauth_sleeps():
    assert not hasattr(webauth, "WRONG_TOKEN_DELAY")
    assert "sleep" not in Path(webauth.__file__).read_text(encoding="utf-8")


def test_five_wrong_bearers_lock_the_ip_even_for_the_right_token(secured, clock):
    fail_n(secured, 5)
    for headers in (BEARER, {"Authorization": "Bearer wrong"}):
        response = secured.get("/projects", headers=headers)
        assert response.status_code == 429
        assert response.get_json() == LOCKED
    clock.now += 59
    assert secured.get("/projects", headers=BEARER).status_code == 429
    clock.now += 2
    assert secured.get("/projects", headers=BEARER).status_code == 200


def test_four_failures_do_not_lock(secured, clock):
    fail_n(secured, 4)
    assert secured.get("/projects", headers=BEARER).status_code == 200


def test_wrong_cookies_and_wrong_logins_count(secured, clock):
    secured.set_cookie("ba_session", "0" * 64)
    secured.get("/projects")
    secured.get("/projects")
    secured.delete_cookie("ba_session")
    secured.post("/login", json={"token": "wrong"})
    secured.post("/login", json={"token": "wrong"})
    secured.post("/login", json={"token": "wrong"})
    assert secured.post("/login", json={"token": TOKEN}).status_code == 429
    assert secured.get("/projects", headers=BEARER).status_code == 429


def test_correct_requests_do_not_count(secured, clock):
    for _ in range(20):
        assert secured.get("/projects", headers=BEARER).status_code == 200
    assert secured.post("/login", json={"token": TOKEN}).status_code == 204


def test_requests_without_credentials_do_not_count(secured, clock):
    for _ in range(10):
        assert secured.get("/projects").status_code == 401
    assert secured.get("/projects", headers=BEARER).status_code == 200


def test_other_ips_are_unaffected(secured, clock):
    fail_n(secured, 5)
    other = {"REMOTE_ADDR": "10.9.8.7"}
    response = secured.get("/projects", headers=BEARER, environ_overrides=other)
    assert response.status_code == 200
    forwarded = {"X-Forwarded-For": "10.9.8.7"}
    assert secured.get("/projects", headers={**BEARER, **forwarded}).status_code == 429


def test_a_lock_expires_failure_by_failure(secured, clock):
    fail_n(secured, 3)
    clock.now += 40
    fail_n(secured, 2)
    assert secured.get("/projects", headers=BEARER).status_code == 429
    clock.now += 21
    assert secured.get("/projects", headers=BEARER).status_code == 200


def test_the_limiter_is_bounded_and_drops_the_oldest(clock):
    limiter = webauth.FailureLimiter(clock=clock, max_ips=3)
    for ip in ("a", "b", "c", "d"):
        limiter.fail(ip)
        clock.now += 1
    assert limiter.tracked() == 3
    assert "a" not in limiter.ips()


def test_expired_entries_are_pruned(clock):
    limiter = webauth.FailureLimiter(clock=clock)
    limiter.fail("a")
    clock.now += 61
    limiter.fail("b")
    assert limiter.ips() == ["b"]
    assert not limiter.blocked("a")
