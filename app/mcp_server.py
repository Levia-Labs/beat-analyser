"""The MCP service: the tools in mcp_tools over Streamable HTTP at /mcp (docs/mcp.md).

Its own compose service, from the web app's image, with uploads/ mounted read-only. Started
as `uvicorn mcp_server:app --workers 1`: jobs live in this process's memory. Stateless HTTP
with JSON responses: no state is kept per session, so any client, curl included, can call a
tool with one POST after initialize.
"""

import hmac
import os

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from netutil import is_loopback

import mcp_tools

TOOLS = (
    mcp_tools.analyse_youtube,
    mcp_tools.analyse_audio,
    mcp_tools.project_status,
    mcp_tools.list_projects,
    mcp_tools.get_events,
)

server = MCPServer(
    "beat-analyser",
    instructions=(
        "Turns audio into timed percussion, beat and note events. Start an analysis with "
        "analyse_audio or analyse_youtube, poll project_status until it is done, then read "
        "events with get_events."
    ),
)
for tool in TOOLS:
    server.tool()(tool)


def body_limit(max_bytes: int) -> int:
    """The request body cap for an upload of up to max_bytes. The SDK default is 4 MB, which
    would refuse any real song: analyse_audio carries the audio as base64 inside JSON, which
    is 4/3 of the size, plus line breaks, plus 1 MB for the JSON-RPC envelope."""
    return max_bytes * 3 // 2 + 1024 * 1024


class BearerToken:
    """ASGI middleware: every HTTP request carries `Authorization: Bearer <token>`."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            given = dict(scope["headers"]).get(b"authorization", b"")
            # Constant time: a byte-by-byte compare leaks the token's prefix through timing.
            if not hmac.compare_digest(given, self.expected):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send(
                    {"type": "http.response.body", "body": b'{"error": "Missing or wrong token"}'}
                )
                return
        await self.app(scope, receive, send)


def create_app(token: str, bind: str, port: int):
    """The ASGI app. Without a token it is loopback only, and Host and Origin are checked
    against DNS rebinding. With one, the token is the check: a rebinding page cannot know it,
    and the Host a remote agent sends is not known in advance (docs/adr/0012)."""
    limit = body_limit(mcp_tools.MAX_BYTES)
    if not token:
        if not is_loopback(bind):
            raise SystemExit(
                f"BIND={bind} publishes the MCP port beyond this machine: set MCP_TOKEN "
                "(docs/mcp.md)"
            )
        security = TransportSecuritySettings(
            allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"],
            allowed_origins=[f"http://127.0.0.1:{port}", f"http://localhost:{port}"],
        )
        return server.streamable_http_app(
            transport_security=security,
            stateless_http=True,
            json_response=True,
            max_request_body_size=limit,
        )
    # The token replaces the Host check: a rebinding page cannot know it, and a remote
    # agent's Host is not known in advance.
    inner = server.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        stateless_http=True,
        json_response=True,
        max_request_body_size=limit,
    )
    return BearerToken(inner, token)


# MCP_PORT is the published port: the Host header an agent sends carries it, not 8001.
app = create_app(
    os.environ.get("MCP_TOKEN", ""),
    os.environ.get("BIND", "127.0.0.1"),
    int(os.environ.get("MCP_PORT", "8001")),
)
