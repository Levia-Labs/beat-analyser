"""The SPA part of tasks/dev/e2e.py: index served, its first asset loads."""

import importlib.util
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


def test_a_missing_asset_fails():
    pages = {"http://h/": (200, INDEX)}
    assert e2e.check_spa("http://h", fake(pages)) == ["/assets/index-a1.js answered 404"]
