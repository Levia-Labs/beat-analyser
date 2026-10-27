"""Helpers of the MCP end-to-end script: reading tool replies, polling to completion."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "dev" / "mcp_e2e.py"
spec = importlib.util.spec_from_file_location("mcp_e2e", SCRIPT)
assert spec and spec.loader
e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e)


def reply(result):
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def test_structured_content_is_the_data():
    got = e2e.tool_result(
        reply({"content": [{"type": "text", "text": "{}"}], "structuredContent": {"name": "x"}})
    )
    assert got == {"error": False, "data": {"name": "x"}, "text": "{}"}


def test_json_text_is_the_data_without_structured_content():
    got = e2e.tool_result(reply({"content": [{"type": "text", "text": '{"name": "x"}'}]}))
    assert got["data"] == {"name": "x"}


def test_a_tool_error_is_an_error():
    got = e2e.tool_result(
        reply({"content": [{"type": "text", "text": "Invalid project name"}], "isError": True})
    )
    assert got == {"error": True, "data": None, "text": "Invalid project name"}


def test_a_protocol_error_is_an_error():
    got = e2e.tool_result({"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "bad"}})
    assert got == {"error": True, "data": None, "text": "bad"}


def test_session_sends_the_bearer_header():
    assert e2e.Session("http://h/mcp", "tok").headers["Authorization"] == "Bearer tok"
    assert "Authorization" not in e2e.Session("http://h/mcp", None).headers


def test_the_token_defaults_to_the_environment(monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    monkeypatch.setenv("WEB_TOKEN", "from-env")
    assert e2e.parse_args(["f.mp3"]).token == "from-env"  # noqa: S105
    monkeypatch.setenv("MCP_TOKEN", "mcp-env")
    assert e2e.parse_args(["f.mp3"]).token == "mcp-env"  # noqa: S105
    assert e2e.parse_args(["f.mp3", "--token", "flag"]).token == "flag"  # noqa: S105


def test_wait_done_polls_until_done():
    states = iter([{"status": "queued"}, {"status": "running"}, {"status": "done"}])
    sleeps = []
    got = e2e.wait_done(
        lambda tool, args: {"data": next(states)}, "x", 60, sleep=sleeps.append, clock=lambda: 0.0
    )
    assert got == {"status": "done"}
    assert sleeps == [2, 2]


def test_wait_done_gives_up_at_the_deadline():
    ticks = iter([0.0, 10.0])
    got = e2e.wait_done(
        lambda tool, args: {"data": {"status": "running"}},
        "x",
        5,
        sleep=lambda seconds: None,
        clock=lambda: next(ticks),
    )
    assert got["status"] == "timeout"
