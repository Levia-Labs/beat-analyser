#!/usr/bin/env python3
"""End-to-end check of the MCP service: analyse a clip through it, poll, read events, check a
refusal. Standard library only, like e2e.py: JSON-RPC over Streamable HTTP, JSON responses.

Usage:
  python3 tasks/dev/mcp_e2e.py FILE [--url http://127.0.0.1:8001/mcp] [--token T] [--timeout 1800]

Sends FILE (name it `fixture-*`) with analyse_audio, polls project_status until done, expects
beat events from get_events, then expects a tool error for the project name `../etc`.
"""

import argparse
import base64
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Offered at initialize; the server answers with the version it speaks, which is then sent.
PROTOCOL = "2025-06-18"
_ids = itertools.count(1)


def rpc(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": next(_ids), "method": method, "params": params}


def tool_result(reply: dict) -> dict:
    """{"error": bool, "data": the tool's dict or None, "text": its first text block}."""
    if "error" in reply:
        return {"error": True, "data": None, "text": reply["error"].get("message", "")}
    result = reply["result"]
    texts = [b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"]
    text = texts[0] if texts else ""
    if result.get("isError"):
        return {"error": True, "data": None, "text": text}
    data = result.get("structuredContent")
    if data is None:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    return {"error": False, "data": data, "text": text}


def wait_done(
    call: Callable[[str, dict], dict],
    name: str,
    timeout: float,
    sleep: Callable[[float], object] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Poll project_status until done, failed or unknown, or timeout seconds pass."""
    deadline = clock() + timeout
    while True:
        state = call("project_status", {"name": name})["data"] or {}
        if state.get("status") in ("done", "failed", "unknown"):
            return state
        if clock() >= deadline:
            return {**state, "status": "timeout"}
        sleep(2)


class Session:
    def __init__(self, url: str, token: str | None) -> None:
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def post(self, message: dict) -> tuple[int, dict | None]:
        req = urllib.request.Request(
            self.url, data=json.dumps(message).encode(), headers=self.headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()
        return status, json.loads(body) if body.strip() else None

    def initialize(self) -> None:
        params = {
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "mcp-e2e", "version": "1"},
        }
        status, reply = self.post(rpc("initialize", params))
        if status != 200 or not reply or "result" not in reply:
            raise SystemExit(f"FAIL  initialize: HTTP {status} {reply}")
        self.headers["MCP-Protocol-Version"] = reply["result"]["protocolVersion"]
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, tool: str, arguments: dict) -> dict:
        status, reply = self.post(rpc("tools/call", {"name": tool, "arguments": arguments}))
        if status != 200 or not reply:
            raise SystemExit(f"FAIL  {tool}: HTTP {status} {reply}")
        return tool_result(reply)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001/mcp")
    parser.add_argument("--token")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    session = Session(args.url, args.token)
    session.initialize()
    failures = []

    data = base64.b64encode(args.file.read_bytes()).decode()
    started = session.call("analyse_audio", {"filename": args.file.name, "data_base64": data})
    if started["error"] or not started["data"]:
        print(f"FAIL  analyse_audio: {started['text']}")
        return 1
    name = started["data"]["name"]
    begun = time.monotonic()
    state = wait_done(session.call, name, args.timeout)
    if state.get("status") != "done":
        print(f"FAIL  {name}: {state.get('status')} {state.get('error', '')}")
        return 1
    tracks = len(state.get("tracks", []))
    print(f"ok    analysed {name} in {time.monotonic() - begun:.1f}s, {tracks} tracks")

    events = session.call("get_events", {"name": name, "track": "beats"})
    rows = (events["data"] or {}).get("rows", [])
    if events["error"] or not rows:
        failures.append(f"get_events beats: {events['text'] or 'no rows'}")
    else:
        print(f"ok    get_events beats: {len(rows)} rows")

    refused = session.call("get_events", {"name": "../etc", "track": "kick"})
    if refused["error"]:
        print("ok    project name ../etc refused")
    else:
        failures.append("get_events accepted the project name ../etc")

    for failure in failures:
        print(f"FAIL  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
