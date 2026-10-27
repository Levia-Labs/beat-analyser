"""The SPA part of tasks/dev/e2e.py: index served, its first asset loads."""

import importlib.util
import io
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "dev" / "e2e.py"
spec = importlib.util.spec_from_file_location("e2e", SCRIPT)
assert spec and spec.loader
e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e)

INDEX = b'<div id="root"></div><script type="module" src="/assets/index-a1.js"></script>'


def fake(pages):
    return lambda url: pages.get(url, (404, b""))


def test_a_served_spa_passes():
    pages = {"http://h/": (200, INDEX), "http://h/assets/index-a1.js": (200, b"x")}
    assert e2e.check_spa("http://h", fake(pages)) == []


def test_a_missing_build_fails():
    assert e2e.check_spa("http://h", fake({"http://h/": (503, b"{}")})) == [
        "/ answered 503, expected the SPA"
    ]


class Captured:
    def __init__(self):
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        raise e2e.urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"x"))


def test_auth_headers():
    assert e2e.auth_headers("tok") == {"Authorization": "Bearer tok"}
    assert e2e.auth_headers("") == {}
    assert e2e.auth_headers(None) == {}


def test_posts_and_fetches_carry_the_bearer_header(monkeypatch):
    seen = Captured()
    monkeypatch.setattr(e2e.urllib.request, "urlopen", seen)
    assert (
        e2e.request("http://h/x", b"", {"Origin": "http://h", **e2e.auth_headers("tok")})[0] == 401
    )
    assert e2e.fetch("http://h/", e2e.auth_headers("tok"))[0] == 401
    assert [r.get_header("Authorization") for r in seen.requests] == ["Bearer tok", "Bearer tok"]


def test_the_token_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv("WEB_TOKEN", "from-env")
    assert e2e.parse_args(["f.mp3"]).token == "from-env"  # noqa: S105
    assert e2e.parse_args(["f.mp3", "--token", "flag"]).token == "flag"  # noqa: S105


def test_a_missing_asset_fails():
    pages = {"http://h/": (200, INDEX)}
    assert e2e.check_spa("http://h", fake(pages)) == ["/assets/index-a1.js answered 404"]
