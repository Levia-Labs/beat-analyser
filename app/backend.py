"""Client for the web app's own analysis endpoints, used by the MCP service's jobs.

Standard library only: each call is one POST that blocks until the project is built, which
urllib does as well as any client library. The response body (the ZIP) is not read; the
status is the answer.
"""

import json
import urllib.error
import urllib.request
import uuid

import youtube

# gunicorn's worker timeout (app/Dockerfile): a long song takes minutes.
TIMEOUT = 1800


class BackendError(Exception):
    """The web app refused or failed the request, or could not be reached."""


def upload(base_url: str, filename: str, data: bytes) -> None:
    """POST /upload. filename is already sanitised (mcp_tools.check_audio)."""
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    _post(f"{base_url}/upload", body, f"multipart/form-data; boundary={boundary}")


def download_yt(base_url: str, video_id: str) -> None:
    """POST /download_yt with the canonical URL rebuilt from the video ID."""
    body = json.dumps({"url": youtube.canonical_url(video_id)}).encode()
    _post(f"{base_url}/download_yt", body, "application/json")


def _post(url: str, body: bytes, content_type: str) -> None:
    # S310: the URL is BACKEND_URL, set by the operator in compose, never by an agent.
    req = urllib.request.Request(  # noqa: S310
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT):  # noqa: S310
            pass
    except urllib.error.HTTPError as exc:
        raise BackendError(_message(exc.code, exc.read())) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BackendError(f"web app unreachable: {exc}") from None


def _message(status: int, body: bytes) -> str:
    try:
        text = str(json.loads(body)["error"])
    except ValueError, KeyError, TypeError:
        # Not the web app's JSON: a gunicorn or proxy page. Quote the start of it.
        text = body[:200].decode("utf-8", "replace").strip()
    return f"{status}: {text}"
