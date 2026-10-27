"""The token handling of tasks/dev/smoke.py: a Bearer header on every request."""

import importlib.util
import io
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "dev" / "smoke.py"
spec = importlib.util.spec_from_file_location("smoke", SCRIPT)
assert spec and spec.loader
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


SECURITY_HEADERS = {name: "x" for name in smoke.HEADERS}


class Reply(io.BytesIO):
    status = 200
    headers = SECURITY_HEADERS


def test_get_sends_the_bearer_header(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        return Reply(b"[]")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake)
    smoke.get("http://h/folders", "tok")
    smoke.get("http://h/folders", None)
    assert [r.get_header("Authorization") for r in seen] == ["Bearer tok", None]


def test_main_uses_the_flag_then_the_environment(monkeypatch):
    seen = []

    def fake(url, token):
        seen.append(token)
        return 200, SECURITY_HEADERS, b"[]"

    monkeypatch.setattr(smoke, "get", fake)
    monkeypatch.setenv("WEB_TOKEN", "from-env")
    assert smoke.main(["http://h"]) == 0
    assert smoke.main(["http://h", "--token", "flag"]) == 0
    assert seen == ["from-env", "from-env", "flag", "flag"]
