"""The MCP server's wiring: tools served in-process through the SDK client, the bearer
token middleware against a stub ASGI app, the start-up refusal."""

import asyncio
import base64
import json

import pytest
from mcp import Client

import mcp_server
import mcp_tools


def test_every_tool_is_served_and_a_refusal_is_a_tool_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "UPLOAD_ROOT", tmp_path)

    async def session():
        async with Client(mcp_server.server) as client:
            listed = await client.list_tools()
            listing = await client.call_tool("list_projects", {})
            refused = await client.call_tool("get_events", {"name": "../etc", "track": "kick"})
            return {tool.name for tool in listed.tools}, listing, refused

    names, listing, refused = asyncio.run(session())
    assert names == {tool.__name__ for tool in mcp_server.TOOLS}
    assert not listing.is_error
    assert refused.is_error
    # getattr: content is a union of block types and only the text block has .text.
    assert "Invalid project name" in getattr(refused.content[0], "text", "")


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),  # noqa: S104 - the input under test, nothing binds to it
        ("192.168.1.5", False),
        ("example.com", False),
    ],
)
def test_is_loopback(host, expected):
    assert mcp_server.is_loopback(host) is expected


def test_no_token_on_a_public_bind_refuses_to_start():
    with pytest.raises(SystemExit, match="MCP_TOKEN"):
        mcp_server.create_app("", "0.0.0.0", 8001)  # noqa: S104 - the refusal under test


@pytest.fixture
def recorded(monkeypatch):
    seen = []

    def streamable_http_app(**kwargs):
        seen.append(kwargs)
        return inner

    monkeypatch.setattr(mcp_server.server, "streamable_http_app", streamable_http_app)
    return seen


def test_create_app_without_a_token_checks_the_host(recorded):
    mcp_server.create_app("", "127.0.0.1", 8001)
    (kwargs,) = recorded
    assert kwargs["max_request_body_size"] == mcp_server.body_limit(mcp_tools.MAX_BYTES)
    assert kwargs["stateless_http"] is True
    assert kwargs["json_response"] is True
    security = kwargs["transport_security"]
    assert security.allowed_hosts == ["127.0.0.1:8001", "localhost:8001"]


def test_create_app_with_a_token_skips_the_host_check(recorded):
    app = mcp_server.create_app("tok", "0.0.0.0", 8001)  # noqa: S104 - the case under test
    (kwargs,) = recorded
    assert isinstance(app, mcp_server.BearerToken)
    assert kwargs["max_request_body_size"] == mcp_server.body_limit(mcp_tools.MAX_BYTES)
    assert kwargs["stateless_http"] is True
    assert kwargs["json_response"] is True
    assert kwargs["transport_security"].enable_dns_rebinding_protection is False


@pytest.mark.parametrize("max_bytes", [4, 1000, 20 * 1024 * 1024])
def test_body_limit_admits_the_largest_allowed_upload(max_bytes):
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "analyse_audio",
                "arguments": {
                    "filename": "song.wav",
                    "data_base64": base64.encodebytes(b"x" * max_bytes).decode(),
                },
            },
        }
    )
    assert len(body.encode()) <= mcp_server.body_limit(max_bytes)


async def inner(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def call(app, headers):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}
    asyncio.run(app(scope, receive, send))
    return sent[0]["status"]


def test_the_right_token_passes():
    app = mcp_server.BearerToken(inner, "s3cret")
    assert call(app, [(b"authorization", b"Bearer s3cret")]) == 200


@pytest.mark.parametrize(
    "headers",
    [[], [(b"authorization", b"Bearer wrong")], [(b"authorization", b"s3cret")]],
)
def test_a_missing_or_wrong_token_is_401(headers):
    app = mcp_server.BearerToken(inner, "s3cret")
    assert call(app, headers) == 401


def test_lifespan_passes_through_the_token_check():
    seen = []

    async def lifespan_app(scope, receive, send):
        seen.append(scope["type"])

    async def nothing():
        return {}

    async def ignore(message):
        return None

    app = mcp_server.BearerToken(lifespan_app, "s3cret")
    asyncio.run(app({"type": "lifespan"}, nothing, ignore))
    assert seen == ["lifespan"]
