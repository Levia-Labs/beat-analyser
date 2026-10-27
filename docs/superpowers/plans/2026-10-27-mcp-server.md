# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second compose service, `mcp`, that lets agents start analyses, poll them and read
events over MCP Streamable HTTP at `http://127.0.0.1:8001/mcp`.

**Architecture:** The `mcp` service runs the web app's image under uvicorn with `uploads/`
mounted read-only. Analysis tools queue a job that posts the audio to the web app's own
`/upload` or `/download_yt`, so Flask, whose project lock is per process, stays the only
writer. Read tools read the project folders directly.

**Tech Stack:** Python 3.14, `mcp` 2.3.0 (`MCPServer`, formerly `FastMCP`), `uvicorn` 0.54.0,
standard-library `urllib` for the web app calls, pytest with Hypothesis, Docker Compose, Task.

**Spec:** [docs/superpowers/specs/2026-10-27-mcp-server-design.md](../specs/2026-10-27-mcp-server-design.md)

## Global Constraints

1. Python target `py314`, line length 100, ruff rules `E F W I UP B SIM S PTH RUF` (`ruff.toml`).
   A lint waiver carries an inline reason.
2. App modules import each other by bare name (`import storage`); add new ones to
   `known-first-party` in `ruff.toml`.
3. Every name that reaches the file system passes `storage.is_valid_name`; every YouTube URL
   is reduced by `youtube.video_id` first.
4. Text I/O passes `encoding="utf-8"`.
5. Every source file stays under 300 lines.
6. `mcp==2.3.0` and `uvicorn==0.54.0` pinned in `app/requirements.txt`. No other new runtime
   dependency.
7. The MCP service never writes under `uploads/`; its mount is `:ro`.
8. Published on `${BIND:-127.0.0.1}:${MCP_PORT:-8001}`; refuses to start without `MCP_TOKEN`
   when `BIND` is not loopback.
9. Analysis jobs: `ThreadPoolExecutor(max_workers=2)`, at most 8 pending, backend timeout 1800 s.
10. `get_events` returns at most 10000 rows.
11. Tests never touch the network or the real `uploads/`.
12. Do not regenerate `types:baseline` or `coverage:baseline` to green a failure.
13. Gate mid-lane with `task gate T="lint:python:run,test:run -- -k <area>,types:check,sg:check"`;
    `task validate` once at the end.

## Review Focus

1. Base64 wrapped at 76 columns (output of the `base64` command): accepted and decoded to the
   same bytes. Pinned in Task 4 (`test_wrapped_base64_is_accepted`, property
   `test_check_audio_accepts_wrapped_base64`).
2. A filename an agent copied from its own disk (`C:\music\My Song.MP3`): project `My_Song`,
   extension `.mp3`, not a refusal. Pinned in Task 4 (`test_analyse_audio_uploads_under_the_sanitised_name`).
3. Analysing a name again after its job failed: starts a new job instead of returning the
   stale failure forever. Pinned in Task 2 (`test_a_failed_name_can_be_started_again`).
4. `get_events` with `end <= start`, `limit=0` or a negative limit: empty or clamped rows, no
   exception. Pinned in Task 4 (`test_get_events_clamps_the_limit`, `test_get_events_empty_window`).
5. The web app answering with a non-JSON page (gunicorn 502, a proxy error) or not at all:
   the job is `failed` with a readable message. Pinned in Task 3
   (`test_a_non_json_error_page_is_quoted`, `test_an_unreachable_backend_is_a_backend_error`).

---

### Task 1: Dependencies and the shared extension list

**Files:**

- Modify: `app/requirements.txt`
- Modify: `ruff.toml:31-34`
- Modify: `app/storage.py:18-20`
- Modify: `app/app.py:22`, `app/app.py:100`, `app/app.py:109-110`

**Interfaces:**

- Produces: `storage.ALLOWED_EXTENSIONS: set[str]` (moved from `app.ALLOWED_EXTENSIONS`, same
  value). The MCP tools need it without importing the Flask app.

- [ ] **Step 1: Pin the dependencies**

In `app/requirements.txt`, insert in alphabetical position:

```text
# The MCP service (app/mcp_server.py, docs/mcp.md). Its own dependencies resolve at install.
mcp==2.3.0
```

after `MarkupSafe==3.0.3`, and

```text
uvicorn==0.54.0
```

after `scipy==1.17.1`.

- [ ] **Step 2: Register the new first-party modules**

In `ruff.toml`, replace the `known-first-party` list with:

```toml
known-first-party = [
    "analysis", "app", "backend", "beatcore", "cli", "conftest", "jobs", "key", "mcp_server",
    "mcp_tools", "notes", "pipeline", "storage", "youtube",
]
```

- [ ] **Step 3: Move the extension list into storage**

In `app/storage.py`, after `ZIP_SUFFIX = "_tracks.zip"` add:

```python
# Audio types accepted from an upload. Shared by the web app and the MCP tools.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
```

In `app/app.py`, delete line 22 (`ALLOWED_EXTENSIONS = {...}`) and replace the three uses:

```python
    return render_template("index.html", accept=",".join(sorted(storage.ALLOWED_EXTENSIONS)))
```

```python
    if extension not in storage.ALLOWED_EXTENSIONS:
        return error(
            f"Unsupported file type; use {', '.join(sorted(storage.ALLOWED_EXTENSIONS))}", 400
        )
```

- [ ] **Step 4: Install and run the existing suite**

Run: `task deps:setup`
Expected: `mcp` and `uvicorn` installed into `.venv`, exit 0.

Run: `task gate T="lint:python:run,test:run,types:check,sg:check"`
Expected: four `ok` lines. The refactor changes no behaviour; the upload tests cover it.

- [ ] **Step 5: Commit**

```bash
git add app/requirements.txt ruff.toml app/storage.py app/app.py
git commit -m "build: add mcp and uvicorn, share the allowed audio extensions" -- app/requirements.txt ruff.toml app/storage.py app/app.py
```

---

### Task 2: Job registry

**Files:**

- Create: `app/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**

- Produces:
  - `jobs.Job` dataclass: `name: str`, `status: str` (`queued | running | done | failed`),
    `error: str | None`.
  - `jobs.JobsFull(Exception)`.
  - `jobs.Jobs(submit: Callable[[Callable[[], None]], object] | None = None, max_pending: int = 8, workers: int = 2)`.
  - `Jobs.start(name: str, work: Callable[[], None]) -> Job` (raises `JobsFull`).
  - `Jobs.get(name: str) -> Job | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs.py`:

```python
"""The MCP job registry: one job per project name, run off the request thread."""

import threading

import pytest

from jobs import Jobs, JobsFull


def inline(work):
    work()


def test_a_job_runs_and_is_done():
    jobs = Jobs(submit=inline)
    job = jobs.start("song", lambda: None)
    assert job.status == "done"
    assert job.error is None
    assert jobs.get("song") is job


def test_status_is_running_while_the_work_runs():
    jobs = Jobs(submit=inline)
    seen = []

    def work():
        job = jobs.get("song")
        assert job is not None
        seen.append(job.status)

    jobs.start("song", work)
    assert seen == ["running"]


def test_a_failing_job_records_the_error():
    def refused():
        raise RuntimeError("400: Unsupported file type")

    job = Jobs(submit=inline).start("song", refused)
    assert job.status == "failed"
    assert job.error == "400: Unsupported file type"


def test_an_exception_without_a_message_is_named():
    def bare():
        raise ValueError

    jobs = Jobs(submit=inline)
    assert jobs.start("song", bare).error == "ValueError"


def test_a_name_already_pending_returns_the_same_job():
    queued = []
    jobs = Jobs(submit=queued.append)
    first = jobs.start("song", lambda: None)
    second = jobs.start("song", lambda: None)
    assert second is first
    assert first.status == "queued"
    assert len(queued) == 1


def test_a_failed_name_can_be_started_again():
    def refused():
        raise RuntimeError("502: Bad Gateway")

    jobs = Jobs(submit=inline)
    jobs.start("song", refused)
    again = jobs.start("song", lambda: None)
    assert again.status == "done"
    assert jobs.get("song") is again


def test_pending_jobs_are_capped():
    jobs = Jobs(submit=lambda work: None, max_pending=2)
    jobs.start("a", lambda: None)
    jobs.start("b", lambda: None)
    with pytest.raises(JobsFull):
        jobs.start("c", lambda: None)


def test_finished_jobs_do_not_count_against_the_cap():
    jobs = Jobs(submit=inline, max_pending=1)
    jobs.start("a", lambda: None)
    assert jobs.start("b", lambda: None).status == "done"


def test_the_default_pool_runs_jobs_on_another_thread():
    ran = threading.Event()
    threads = []

    def work():
        threads.append(threading.current_thread())
        ran.set()

    Jobs().start("song", work)
    assert ran.wait(5)
    assert threads[0] is not threading.current_thread()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `task test:file F=tests/test_jobs.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'jobs'`.

- [ ] **Step 3: Write the registry**

`app/jobs.py`:

```python
"""Analysis jobs for the MCP service: one per project name, run in a small thread pool.

The MCP service analyses nothing itself. A job hands the audio to the web app's own endpoint
(backend.py), so Flask, whose project lock only holds inside its own process (pipeline.py),
stays the only writer of uploads/ (docs/adr/0012).
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

log = logging.getLogger("beat-analyser.jobs")

PENDING = ("queued", "running")


@dataclass
class Job:
    name: str
    status: str = "queued"
    error: str | None = None


class JobsFull(Exception):
    """More jobs are pending than the registry accepts."""


class Jobs:
    def __init__(
        self,
        submit: Callable[[Callable[[], None]], object] | None = None,
        max_pending: int = 8,
        workers: int = 2,
    ) -> None:
        # Two workers: the web app runs one gunicorn worker with four threads, and each
        # analysis holds one of them for minutes.
        if submit is None:
            submit = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job").submit
        self._submit = submit
        self._max_pending = max_pending
        self._jobs: dict[str, Job] = {}
        self._guard = threading.Lock()

    def start(self, name: str, work: Callable[[], None]) -> Job:
        """Queue work for name, or return the job already queued or running for it."""
        with self._guard:
            current = self._jobs.get(name)
            if current is not None and current.status in PENDING:
                return current
            pending = sum(job.status in PENDING for job in self._jobs.values())
            if pending >= self._max_pending:
                raise JobsFull(f"{pending} analyses pending; try again when one finishes")
            job = Job(name)
            self._jobs[name] = job
        self._submit(lambda: self._run(job, work))
        return job

    def get(self, name: str) -> Job | None:
        with self._guard:
            return self._jobs.get(name)

    @staticmethod
    def _run(job: Job, work: Callable[[], None]) -> None:
        job.status = "running"
        try:
            work()
        except Exception as exc:
            # Recorded, never raised: the pool would swallow it and the agent would poll a
            # job stuck in "running".
            log.exception("job %s failed", job.name)
            job.error = str(exc) or type(exc).__name__
            job.status = "failed"
            return
        job.status = "done"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `task gate T="lint:python:run,test:run -- -k jobs,types:check,sg:check"`
Expected: four `ok` lines.

- [ ] **Step 5: Commit**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat(mcp): add the analysis job registry" -- app/jobs.py tests/test_jobs.py
```

---

### Task 3: Web app client

**Files:**

- Create: `app/backend.py`
- Test: `tests/test_backend.py`

**Interfaces:**

- Consumes: `youtube.canonical_url(video_id: str) -> str` (exists).
- Produces:
  - `backend.TIMEOUT = 1800`.
  - `backend.BackendError(Exception)`; its message is `"<status>: <web app error>"` or
    `"web app unreachable: <reason>"`.
  - `backend.upload(base_url: str, filename: str, data: bytes) -> None`.
  - `backend.download_yt(base_url: str, video_id: str) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_backend.py`:

```python
"""The MCP service's client for the web app's analysis endpoints. urlopen is replaced, so
no request leaves the process."""

import io
import json
import urllib.error
from email.message import Message

import pytest

import backend
import youtube


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    requests = []

    def urlopen(req, timeout):
        requests.append((req, timeout))
        return Response()

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)
    return requests


def refuse_with(monkeypatch, exc):
    def urlopen(req, timeout):
        raise exc

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)


def http_error(status, body):
    return urllib.error.HTTPError(
        "http://backend:8000/upload", status, "error", Message(), io.BytesIO(body)
    )


def test_upload_posts_multipart_with_the_filename(sent):
    backend.upload("http://backend:8000", "song.mp3", b"ID3data")
    req, timeout = sent[0]
    assert req.full_url == "http://backend:8000/upload"
    assert req.get_method() == "POST"
    assert timeout == backend.TIMEOUT
    content_type = req.get_header("Content-type")
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("=", 1)[1]
    assert req.data.startswith(f"--{boundary}\r\n".encode())
    assert b'name="file"; filename="song.mp3"' in req.data
    assert b"\r\n\r\nID3data\r\n" in req.data
    assert req.data.endswith(f"--{boundary}--\r\n".encode())


def test_download_yt_posts_the_canonical_url(sent):
    backend.download_yt("http://backend:8000", "dQw4w9WgXcQ")
    req, _ = sent[0]
    assert req.full_url == "http://backend:8000/download_yt"
    assert req.get_header("Content-type") == "application/json"
    assert json.loads(req.data) == {"url": youtube.canonical_url("dQw4w9WgXcQ")}


def test_a_refusal_carries_the_web_app_message(monkeypatch):
    refuse_with(monkeypatch, http_error(400, b'{"error": "Unsupported file type"}'))
    with pytest.raises(backend.BackendError, match=r"^400: Unsupported file type$"):
        backend.upload("http://backend:8000", "song.mp3", b"x")


def test_a_non_json_error_page_is_quoted(monkeypatch):
    refuse_with(monkeypatch, http_error(502, b"<html>Bad Gateway</html>"))
    with pytest.raises(backend.BackendError, match=r"^502: <html>Bad Gateway</html>$"):
        backend.upload("http://backend:8000", "song.mp3", b"x")


def test_an_unreachable_backend_is_a_backend_error(monkeypatch):
    refuse_with(monkeypatch, urllib.error.URLError("Connection refused"))
    with pytest.raises(backend.BackendError, match="unreachable"):
        backend.download_yt("http://backend:8000", "dQw4w9WgXcQ")


def test_a_timeout_is_a_backend_error(monkeypatch):
    refuse_with(monkeypatch, TimeoutError("timed out"))
    with pytest.raises(backend.BackendError, match="unreachable"):
        backend.upload("http://backend:8000", "song.mp3", b"x")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `task test:file F=tests/test_backend.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'backend'`.

- [ ] **Step 3: Write the client**

`app/backend.py`:

```python
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
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    _post(f"{base_url}/upload", body, f"multipart/form-data; boundary={boundary}")


def download_yt(base_url: str, video_id: str) -> None:
    """POST /download_yt with the canonical URL rebuilt from the video ID."""
    body = json.dumps({"url": youtube.canonical_url(video_id)}).encode()
    _post(f"{base_url}/download_yt", body, "application/json")


def _post(url: str, body: bytes, content_type: str) -> None:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        # S310: the URL is BACKEND_URL, set by the operator in compose, never by an agent.
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `task gate T="lint:python:run,test:run -- -k backend,types:check,sg:check"`
Expected: four `ok` lines.

- [ ] **Step 5: Commit**

```bash
git add app/backend.py tests/test_backend.py
git commit -m "feat(mcp): add the web app client for analysis jobs" -- app/backend.py tests/test_backend.py
```

---

### Task 4: The tools

**Files:**

- Create: `app/mcp_tools.py`
- Test: `tests/test_mcp_tools.py`
- Modify: `tests/test_properties.py` (imports, three properties at the end)

**Interfaces:**

- Consumes: `jobs.Jobs`, `jobs.JobsFull` (Task 2); `backend.upload`, `backend.download_yt`,
  `backend.BackendError` (Task 3); `storage.ALLOWED_EXTENSIONS` (Task 1);
  `storage.project_name`, `storage.is_valid_name`, `storage.project_dir`, `storage.zip_path`,
  `storage.list_projects`, `storage.META_NAME`, `youtube.video_id` (exist);
  `ToolError` from `mcp.server.mcpserver.exceptions`.
- Produces (module attributes tests patch: `UPLOAD_ROOT`, `BACKEND_URL`, `MAX_BYTES`, `jobs`):
  - `check_audio(filename: str, data_base64: str, max_bytes: int) -> tuple[str, str, bytes]` (raises `ValueError`).
  - `analyse_youtube(url: str) -> dict[str, Any]` returns `{name, status}`.
  - `analyse_audio(filename: str, data_base64: str) -> dict[str, Any]` returns `{name, status}`.
  - `project_status(name: str) -> dict[str, Any]`.
  - `list_projects() -> dict[str, Any]` returns `{projects: [{name, done, meta}]}`.
  - `get_events(name: str, track: str, start: float = 0.0, end: float | None = None, limit: int = 1000) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mcp_tools.py`:

```python
"""The MCP tools, called as plain functions: jobs run inline, the web app is a stub, and
projects live in a temporary root."""

import base64
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import backend
import mcp_tools
import storage
from jobs import Jobs


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(mcp_tools, "jobs", Jobs(submit=lambda work: work()))
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(backend, "upload", lambda *args: seen.append(("upload", *args)))
    monkeypatch.setattr(backend, "download_yt", lambda *args: seen.append(("yt", *args)))
    return seen


def make_project(root, name="song", built=True):
    folder = root / name
    folder.mkdir()
    (folder / "beats.csv").write_text("0.5,131\n1,131\n1.5,131\nbad,row\n2,131\n", encoding="utf-8")
    (folder / "kick.csv").write_text("", encoding="utf-8")
    (folder / "labels.json").write_text(json.dumps({"131": "beat"}), encoding="utf-8")
    meta = {"name": name, "key": "A minor", "bpm": 120.0}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if built:
        storage.zip_path(root, name).write_bytes(b"PK")
    return folder


def test_analyse_audio_uploads_under_the_sanitised_name(root, calls):
    result = mcp_tools.analyse_audio("C:\\music\\My Song.MP3", b64(b"ID3data"))
    assert result == {"name": "My_Song", "status": "done"}
    assert calls == [("upload", mcp_tools.BACKEND_URL, "My_Song.mp3", b"ID3data")]


def test_wrapped_base64_is_accepted(root, calls):
    data = bytes(range(256)) * 2
    mcp_tools.analyse_audio("song.wav", base64.encodebytes(data).decode())
    assert calls[0][3] == data


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("notes.txt", b64(b"x"), "Unsupported file type"),
        (".wav", b64(b"x"), "Unsupported file type"),
        ("song.wav", "@@@@", "not valid base64"),
        ("song.wav", "", "Empty audio"),
    ],
)
def test_analyse_audio_refusals(root, calls, filename, payload, message):
    with pytest.raises(ToolError, match=message):
        mcp_tools.analyse_audio(filename, payload)
    assert calls == []


def test_analyse_audio_refuses_more_than_the_limit(root, calls, monkeypatch):
    monkeypatch.setattr(mcp_tools, "MAX_BYTES", 4)
    with pytest.raises(ToolError, match="larger than"):
        mcp_tools.analyse_audio("song.wav", b64(b"12345"))
    assert calls == []


def test_analyse_youtube_reduces_the_url_to_an_id(root, calls):
    result = mcp_tools.analyse_youtube("https://youtu.be/dQw4w9WgXcQ")
    assert result == {"name": "dQw4w9WgXcQ", "status": "done"}
    assert calls == [("yt", mcp_tools.BACKEND_URL, "dQw4w9WgXcQ")]


def test_analyse_youtube_refuses_a_foreign_host(root, calls):
    with pytest.raises(ToolError):
        mcp_tools.analyse_youtube("https://example.com/watch?v=dQw4w9WgXcQ")
    assert calls == []


def test_a_full_registry_is_a_tool_error(root, monkeypatch, calls):
    monkeypatch.setattr(mcp_tools, "jobs", Jobs(submit=lambda work: None, max_pending=1))
    mcp_tools.analyse_audio("a.wav", b64(b"x"))
    with pytest.raises(ToolError, match="pending"):
        mcp_tools.analyse_audio("b.wav", b64(b"x"))


def test_a_refused_upload_is_a_failed_job(root, monkeypatch):
    def refuse(*args):
        raise backend.BackendError("400: Unsupported file type")

    monkeypatch.setattr(backend, "upload", refuse)
    mcp_tools.analyse_audio("song.wav", b64(b"x"))
    assert mcp_tools.project_status("song") == {
        "name": "song",
        "status": "failed",
        "error": "400: Unsupported file type",
    }


def test_a_project_built_by_the_web_ui_is_done(root):
    make_project(root)
    status = mcp_tools.project_status("song")
    assert status["status"] == "done"
    assert status["meta"]["key"] == "A minor"
    assert status["tracks"] == [
        {"track": "beats", "events": 4, "first": 0.5, "last": 2.0},
        {"track": "kick", "events": 0, "first": None, "last": None},
    ]


def test_a_folder_without_its_zip_is_unknown(root):
    make_project(root, built=False)
    assert mcp_tools.project_status("song") == {"name": "song", "status": "unknown"}


@pytest.mark.parametrize("tool", ["project_status", "get_events"])
def test_a_traversal_name_is_refused(root, tool):
    args = ("../etc",) if tool == "project_status" else ("../etc", "kick")
    with pytest.raises(ToolError, match="Invalid project name"):
        getattr(mcp_tools, tool)(*args)


def test_list_projects(root):
    make_project(root)
    (root / "half").mkdir()
    assert mcp_tools.list_projects() == {
        "projects": [
            {"name": "half", "done": False, "meta": None},
            {"name": "song", "done": True, "meta": {"name": "song", "key": "A minor", "bpm": 120.0}},
        ]
    }


def test_list_projects_without_a_root(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "UPLOAD_ROOT", tmp_path / "missing")
    assert mcp_tools.list_projects() == {"projects": []}


def test_get_events_window(root):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", start=1, end=2)
    assert result == {
        "name": "song",
        "track": "beats",
        "rows": [[1.0, "beat"], [1.5, "beat"]],
        "truncated": False,
    }


def test_get_events_truncates_at_the_limit(root):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", limit=2)
    assert result["rows"] == [[0.5, "beat"], [1.0, "beat"]]
    assert result["truncated"] is True


@pytest.mark.parametrize("limit", [0, -5])
def test_get_events_clamps_the_limit(root, limit):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", limit=limit)
    assert result["rows"] == [[0.5, "beat"]]
    assert result["truncated"] is True


def test_get_events_empty_window(root):
    make_project(root)
    result = mcp_tools.get_events("song", "beats", start=2, end=1)
    assert result["rows"] == []
    assert result["truncated"] is False


def test_get_events_without_labels_returns_the_id(root):
    folder = make_project(root)
    (folder / "labels.json").unlink()
    assert mcp_tools.get_events("song", "beats", limit=1)["rows"] == [[0.5, "131"]]


def test_get_events_refuses_an_unknown_track(root):
    make_project(root)
    with pytest.raises(ToolError, match="tracks: beats, kick"):
        mcp_tools.get_events("song", "../meta")
```

- [ ] **Step 2: Add the properties**

In `tests/test_properties.py`, add `import base64` to the standard-library imports, `import pytest`
after the `hypothesis` imports, `import mcp_tools` between `import analysis` and `import notes`,
and at the end of the file:

```python
EXTENSION = st.sampled_from([*sorted(storage.ALLOWED_EXTENSIONS), ".MP3", ".txt", ""])


@given(FILENAME, EXTENSION, st.binary(max_size=64))
def test_check_audio_returns_the_bytes_or_refuses(stem, extension, data):
    try:
        name, ext, decoded = mcp_tools.check_audio(stem + extension, base64.b64encode(data).decode(), 64)
    except ValueError:
        return
    assert storage.is_valid_name(name)
    assert ext in storage.ALLOWED_EXTENSIONS
    assert decoded == data


@given(st.binary(min_size=65, max_size=300), st.booleans())
def test_check_audio_refuses_more_than_the_limit(data, wrapped):
    encoded = base64.encodebytes(data) if wrapped else base64.b64encode(data)
    with pytest.raises(ValueError):
        mcp_tools.check_audio("a.wav", encoded.decode(), 64)


@given(st.binary(min_size=1, max_size=300))
def test_check_audio_accepts_wrapped_base64(data):
    assert mcp_tools.check_audio("a.wav", base64.encodebytes(data).decode(), 300)[2] == data
```

- [ ] **Step 3: Run them to verify they fail**

Run: `task test:file F=tests/test_mcp_tools.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'mcp_tools'`.

- [ ] **Step 4: Write the tools**

`app/mcp_tools.py`:

```python
"""The MCP tools: start an analysis, poll it, read what it produced (docs/mcp.md).

Plain functions, registered on the server in mcp_server.py, so tests call them directly.
Every argument is checked here before it reaches a job or the file system: names through
storage.is_valid_name, URLs through youtube.video_id, audio through check_audio. Nothing here
writes under UPLOAD_ROOT; the service mounts it read-only.
"""

import base64
import binascii
import csv
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

import backend
import storage
import youtube
from jobs import Jobs, JobsFull

MAX_ROWS = 10000
# analysis.LABELS_NAME. Not imported: analysis loads onnxruntime and the note model.
LABELS_NAME = "labels.json"

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "uploads")).resolve()
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024

jobs = Jobs()


def check_audio(filename: str, data_base64: str, max_bytes: int) -> tuple[str, str, bytes]:
    """Project name, extension and bytes of an agent's upload. Raises ValueError.

    The length is checked before decoding, so an oversized payload is refused without a
    second copy in memory. Line breaks are allowed: the `base64` command wraps at 76.
    """
    extension = Path(filename.replace("\\", "/")).suffix.lower()
    if extension not in storage.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(storage.ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type; use {allowed}")
    name = storage.project_name(filename)
    limit = 4 * ((max_bytes + 2) // 3)
    # At most two line-break characters per 76 of data, plus a trailing one.
    if len(data_base64) > limit + limit // 38 + 2:
        raise ValueError(f"Audio larger than {max_bytes} bytes")
    compact = "".join(data_base64.split())
    if len(compact) > limit:
        raise ValueError(f"Audio larger than {max_bytes} bytes")
    try:
        data = base64.b64decode(compact, validate=True)
    except binascii.Error:
        raise ValueError("data_base64 is not valid base64") from None
    if not data:
        raise ValueError("Empty audio")
    if len(data) > max_bytes:
        raise ValueError(f"Audio larger than {max_bytes} bytes")
    return name, extension, data


def analyse_youtube(url: str) -> dict[str, Any]:
    """Start analysing a YouTube video. Returns at once with the project name (the video ID);
    poll project_status(name) until it is done or failed."""
    try:
        vid = youtube.video_id(url)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    return _start(vid, lambda: backend.download_yt(BACKEND_URL, vid))


def analyse_audio(filename: str, data_base64: str) -> dict[str, Any]:
    """Start analysing an audio file (wav, mp3, flac, ogg, m4a) sent as base64. Returns at
    once with the project name (the file's sanitised stem); poll project_status(name)."""
    try:
        name, extension, data = check_audio(filename, data_base64, MAX_BYTES)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    return _start(name, lambda: backend.upload(BACKEND_URL, f"{name}{extension}", data))


def project_status(name: str) -> dict[str, Any]:
    """State of a project: queued, running, failed (with error), done (with meta and the event
    count and time span of each track), or unknown."""
    _check_name(name)
    job = jobs.get(name)
    if job is not None and job.status != "done":
        result: dict[str, Any] = {"name": name, "status": job.status}
        if job.error:
            result["error"] = job.error
        return result
    # Only the ZIP means done: pipeline.build writes meta.json before it packages.
    if not storage.zip_path(UPLOAD_ROOT, name).is_file():
        return {"name": name, "status": "unknown"}
    folder = storage.project_dir(UPLOAD_ROOT, name)
    return {"name": name, "status": "done", "meta": _meta(folder), "tracks": _summary(folder)}


def list_projects() -> dict[str, Any]:
    """Every project: its name, whether its ZIP is built, and meta.json (key, bpm)."""
    if not UPLOAD_ROOT.is_dir():
        return {"projects": []}
    projects = [
        {
            "name": name,
            "done": storage.zip_path(UPLOAD_ROOT, name).is_file(),
            "meta": files.get(storage.META_NAME, {}).get("content"),
        }
        for name, files in storage.list_projects(UPLOAD_ROOT).items()
    ]
    return {"projects": projects}


def get_events(
    name: str, track: str, start: float = 0.0, end: float | None = None, limit: int = 1000
) -> dict[str, Any]:
    """Events of one track as [time in seconds, label] rows with start <= time < end, at most
    limit rows (up to 10000). truncated says whether more rows matched."""
    _check_name(name)
    folder = storage.project_dir(UPLOAD_ROOT, name)
    tracks = _tracks(folder)
    if track not in tracks:
        listed = ", ".join(tracks) or "none"
        raise ToolError(f"No track '{track}' in project '{name}'; tracks: {listed}")
    labels = _labels(folder)
    limit = max(1, min(limit, MAX_ROWS))
    rows: list[list[Any]] = []
    truncated = False
    for time, label_id in _rows(tracks[track]):
        if time < start or (end is not None and time >= end):
            continue
        if len(rows) == limit:
            truncated = True
            break
        rows.append([time, labels.get(label_id, str(label_id))])
    return {"name": name, "track": track, "rows": rows, "truncated": truncated}


def _start(name: str, work: Callable[[], None]) -> dict[str, Any]:
    try:
        job = jobs.start(name, work)
    except JobsFull as exc:
        raise ToolError(str(exc)) from None
    return {"name": name, "status": job.status}


def _check_name(name: str) -> None:
    if not storage.is_valid_name(name):
        raise ToolError(f"Invalid project name '{name[:100]}'")


def _tracks(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {path.stem: path for path in sorted(folder.glob("*.csv"))}


def _rows(path: Path) -> Iterator[tuple[float, int]]:
    """(time, label ID) per CSV row. A malformed row is skipped: this reads, it does not
    validate."""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            try:
                yield float(row[0]), int(row[1])
            except IndexError, ValueError:
                continue


def _summary(folder: Path) -> list[dict[str, Any]]:
    summary = []
    for track, path in _tracks(folder).items():
        times = [time for time, _ in _rows(path)]
        first, last = (times[0], times[-1]) if times else (None, None)
        summary.append({"track": track, "events": len(times), "first": first, "last": last})
    return summary


def _labels(folder: Path) -> dict[int, str]:
    try:
        raw = json.loads((folder / LABELS_NAME).read_text(encoding="utf-8"))
        return {int(key): str(value) for key, value in raw.items()}
    except OSError, ValueError, AttributeError:
        return {}


def _meta(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / storage.META_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `task gate T="lint:python:run,test:run -- tests/test_mcp_tools.py tests/test_properties.py,types:check,sg:check"`
Expected: four `ok` lines.

- [ ] **Step 6: Commit**

```bash
git add app/mcp_tools.py tests/test_mcp_tools.py tests/test_properties.py
git commit -m "feat(mcp): add the analysis, status and event tools" -- app/mcp_tools.py tests/test_mcp_tools.py tests/test_properties.py
```

---

### Task 5: The server

**Files:**

- Create: `app/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**

- Consumes: the five tool functions from `mcp_tools` (Task 4); `MCPServer` from `mcp.server`;
  `TransportSecuritySettings` from `mcp.server.transport_security`; `Client` from `mcp`.
- Produces:
  - `mcp_server.TOOLS: tuple` (the registered functions), `mcp_server.server: MCPServer`.
  - `mcp_server.is_loopback(host: str) -> bool`.
  - `mcp_server.BearerToken(app, token: str)`: ASGI middleware.
  - `mcp_server.create_app(token: str, bind: str, port: int)`: the ASGI app; raises
    `SystemExit` without a token on a non-loopback `bind`.
  - `mcp_server.app`: what `uvicorn mcp_server:app` serves, at path `/mcp`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mcp_server.py`:

```python
"""The MCP server's wiring: tools served in-process through the SDK client, the bearer
token middleware against a stub ASGI app, the start-up refusal."""

import asyncio

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
    [("127.0.0.1", True), ("::1", True), ("localhost", True), ("0.0.0.0", False),
     ("192.168.1.5", False), ("example.com", False)],
)
def test_is_loopback(host, expected):
    assert mcp_server.is_loopback(host) is expected


def test_no_token_on_a_public_bind_refuses_to_start():
    with pytest.raises(SystemExit, match="MCP_TOKEN"):
        mcp_server.create_app("", "0.0.0.0", 8001)


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `task test:file F=tests/test_mcp_server.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'mcp_server'`.

- [ ] **Step 3: Write the server**

`app/mcp_server.py`:

```python
"""The MCP service: the tools in mcp_tools over Streamable HTTP at /mcp (docs/mcp.md).

Its own compose service, from the web app's image, with uploads/ mounted read-only. Started
as `uvicorn mcp_server:app --workers 1`: jobs live in this process's memory. Stateless HTTP
with JSON responses: no state is kept per session, so any client, curl included, can call a
tool with one POST after initialize.
"""

import hmac
import ipaddress
import os

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

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


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
            transport_security=security, stateless_http=True, json_response=True
        )
    # S104: a non-loopback host turns off the SDK's automatic loopback-only Host check,
    # which would refuse a remote agent; the token replaces it.
    inner = server.streamable_http_app(
        host="0.0.0.0",  # noqa: S104
        stateless_http=True,
        json_response=True,
    )
    return BearerToken(inner, token)


# MCP_PORT is the published port: the Host header an agent sends carries it, not 8001.
app = create_app(
    os.environ.get("MCP_TOKEN", ""),
    os.environ.get("BIND", "127.0.0.1"),
    int(os.environ.get("MCP_PORT", "8001")),
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `task gate T="lint:python:run,test:run -- -k mcp_server,types:check,sg:check"`
Expected: four `ok` lines. If the in-process test fails on an SDK name (`Client`,
`list_tools().tools`, `is_error`), check the installed package with
`task sg:search P='class Client($$$A)' -- .venv` and fix the name in the test and the module.
The spec's facts come from the 2.x migration guide.

- [ ] **Step 5: Commit**

```bash
git add app/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): serve the tools over Streamable HTTP with a token option" -- app/mcp_server.py tests/test_mcp_server.py
```

---

### Task 6: Compose service and the end-to-end check

**Files:**

- Modify: `docker-compose.yml` (new `mcp` service after `backend`)
- Create: `tasks/dev/mcp_e2e.py`
- Test: `tests/test_mcp_e2e.py`
- Modify: `tasks/dev/Taskfile.yml` (new `mcp:e2e` after `e2e`)
- Modify: `tasks/docker/Taskfile.yml:18-22` (`up`) and `:70-84` (`e2e`)
- Modify: `tasks/dev/README.md` (row after `e2e.py`), `docs/tasks.md` (row after `dev:e2e`)

**Interfaces:**

- Consumes: the `app` from Task 5 at `/mcp`; `task dev:fixture`; `task uploads:fixtures:clean`.
- Produces:
  - `task dev:mcp:e2e [URL=http://127.0.0.1:8001/mcp] [TOKEN=]`.
  - `mcp_e2e.tool_result(reply: dict) -> dict` with keys `error: bool`, `data: dict | None`, `text: str`.
  - `mcp_e2e.wait_done(call, name, timeout, sleep=time.sleep, clock=time.monotonic) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mcp_e2e.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `task test:file F=tests/test_mcp_e2e.py`
Expected: FAIL, `FileNotFoundError` for `tasks/dev/mcp_e2e.py`.

- [ ] **Step 3: Write the script**

`tasks/dev/mcp_e2e.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `task gate T="lint:python:run,hostpy:run,test:run -- -k mcp_e2e,types:check"`
Expected: four `ok` lines.

- [ ] **Step 5: Add the compose service**

In `docker-compose.yml`, after the `backend` service's `restart: "no"`, add:

```yaml
# MCP for agents (docs/mcp.md). Same image, its own process. uploads/ is read-only here:
# analyses go through the backend's endpoints, whose project lock is per process
# (docs/adr/0012).
mcp:
  container_name: beat-analyser-mcp
  image: beat-analyser:latest
  # Built by the backend service; never pulled.
  pull_policy: never
  command: ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
  ports:
    - "${BIND:-127.0.0.1}:${MCP_PORT:-8001}:8001"
  environment:
    BACKEND_URL: http://backend:8000
    BIND: "${BIND:-127.0.0.1}"
    MCP_PORT: "${MCP_PORT:-8001}"
    MCP_TOKEN: "${MCP_TOKEN:-}"
    MAX_UPLOAD_MB: "${MAX_UPLOAD_MB:-200}"
  volumes:
    - ./uploads:/app/uploads:ro
  read_only: true
  tmpfs:
    - /tmp
  cap_drop:
    - ALL
  security_opt:
    - no-new-privileges:true
  depends_on:
    backend:
      condition: service_healthy
  # A TCP connect, not an HTTP call: with MCP_TOKEN set, an unauthenticated request is 401.
  healthcheck:
    test:
      [
        "CMD",
        "python",
        "-c",
        "import socket; socket.create_connection(('127.0.0.1', 8001), 5).close()",
      ]
    interval: 30s
    timeout: 10s
    retries: 3
  restart: "no"
```

- [ ] **Step 6: Add the tasks**

In `tasks/dev/Taskfile.yml`, after the `e2e` task:

```yaml
mcp:e2e:
  desc: "Analyse a fixture through a running MCP service and check status, events and a refusal. Usage: task dev:mcp:e2e [URL=http://127.0.0.1:8001/mcp] [TOKEN=]"
  cmds:
    # A cached ZIP would make the project done at once, and the check would pass on stale
    # output. Fixture projects only; nothing else is touched.
    - task: :uploads:fixtures:clean
    - task: fixture
      vars: { NAME: fixture-mcp }
    - '"{{.SYS_PY}}" tasks/dev/mcp_e2e.py "{{.DEV_FIXTURES}}/fixture-mcp.mp3" --url "{{.URL | default "http://127.0.0.1:8001/mcp"}}"{{if .TOKEN}} --token "{{.TOKEN}}"{{end}}'
```

In `tasks/docker/Taskfile.yml`, replace the `up` task's echo line with two:

```yaml
- echo "http://localhost:${PORT:-80}  (task docker:logs to follow)"
- echo "MCP  http://localhost:${MCP_PORT:-8001}/mcp  (docs/mcp.md)"
```

and its `desc` with:

```yaml
desc: "Start the service and its MCP endpoint in the background. Usage: task docker:up [PORT=80] [MCP_PORT=8001] [BIND=127.0.0.1]"
```

Replace the `e2e` task with:

```yaml
e2e:
  desc: "Build, start on spare ports, run task dev:e2e and dev:mcp:e2e against it, stop. Usage: task docker:e2e [PORT=8099] [MCP_PORT=8098]"
  vars:
    PORT: '{{.PORT | default "8099"}}'
    MCP_PORT: '{{.MCP_PORT | default "8098"}}'
  env:
    PORT: "{{.PORT}}"
    MCP_PORT: "{{.MCP_PORT}}"
  cmds:
    - defer: docker compose down
    # --wait blocks until both healthchecks pass, so no sleep and no retry loop.
    - docker compose up -d --build --wait
    - task: :dev:e2e
      vars: { URL: "http://127.0.0.1:{{.PORT}}" }
    - task: :summary
      vars: { NAME: fixture-e2e }
    # After the summary: dev:mcp:e2e clears fixture projects before its own run.
    - task: :dev:mcp:e2e
      vars: { URL: "http://127.0.0.1:{{.MCP_PORT}}/mcp" }
    - docker images beat-analyser --format "image {{"{{"}}.Size{{"}}"}}"
```

- [ ] **Step 7: Index the new task**

`tasks/dev/README.md`, after the `e2e.py` row (pad to the table's column widths with
`task lint:run` afterwards):

```markdown
| `mcp_e2e.py` | `task dev:mcp:e2e [URL=] [TOKEN=]`, `task docker:e2e` | Analyse a fixture through the MCP service, poll it, read beat events, check a name refusal |
```

`docs/tasks.md`, after the `dev:e2e` row:

```markdown
| `dev:mcp:e2e [URL=] [TOKEN=]` | Fixture analysis, status, events and a refusal through a running MCP service | local |
```

- [ ] **Step 8: Verify the container**

Run: `task gate T="docker:config,lint:yaml:check,quality:taskfiles"`
Expected: three `ok` lines.

Run: `task gate T=docker:e2e`
Expected: `ok    docker:e2e`. The log in `.cache/gate/` shows `ok    analysed fixture-mcp`,
`ok    get_events beats`, `ok    project name ../etc refused`. If it fails, read the decisive
lines; the likely ones are an SDK name (Task 5 Step 4) or a 421 or 403 from the Host check
(compare `MCP_PORT` in the container with the Host the script sends).

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml tasks/dev/mcp_e2e.py tests/test_mcp_e2e.py tasks/dev/Taskfile.yml tasks/docker/Taskfile.yml tasks/dev/README.md docs/tasks.md
git commit -m "feat(mcp): add the mcp compose service and its end-to-end check" -- docker-compose.yml tasks/dev/mcp_e2e.py tests/test_mcp_e2e.py tasks/dev/Taskfile.yml tasks/docker/Taskfile.yml tasks/dev/README.md docs/tasks.md
```

---

### Task 7: Documentation

**Files:**

- Create: `docs/mcp.md`
- Create: `docs/adr/0012-mcp-as-a-sibling-proxy.md`
- Modify: `docs/adr/README.md` (row after 0011), `docs/README.md` (row after `api.md`),
  `app/README.md` (rows after `youtube.py`), `docs/configuration.md` (rows after `HOST`),
  `docs/security.md` (rows at the end of the "What is protected" table, a paragraph in
  "Before setting `BIND=0.0.0.0`"), `docs/architecture.md` (a section on the second service),
  `README.md` (one line where the API doc is linked)

- [ ] **Step 1: Write `docs/mcp.md`**

````markdown
# MCP

Agents connect to the service over the Model Context Protocol (Streamable HTTP). The `mcp`
compose service starts with `task docker:up` and listens on `http://127.0.0.1:8001/mcp`.

## Connect

Claude Code, in `.mcp.json` or with `claude mcp add --transport http beat-analyser http://127.0.0.1:8001/mcp`:

```json
{ "mcpServers": { "beat-analyser": { "type": "http", "url": "http://127.0.0.1:8001/mcp" } } }
```

With `MCP_TOKEN` set, add `"headers": { "Authorization": "Bearer <token>" }`.

Analyses take minutes, so they run as jobs: start one, then poll.

## Tools

| Tool              | Arguments                                       | Returns                                                                                             |
| ----------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `analyse_audio`   | `filename`, `data_base64`                       | `{name, status}` at once. Same types and size limit as `POST /upload`                               |
| `analyse_youtube` | `url`                                           | `{name, status}` at once. Same checks as `POST /download_yt`                                        |
| `project_status`  | `name`                                          | `status`: `queued`, `running`, `failed` with `error`, `done` with `meta` and `tracks`, or `unknown` |
| `list_projects`   | none                                            | `{projects: [{name, done, meta}]}`                                                                  |
| `get_events`      | `name`, `track`, `start=0`, `end`, `limit=1000` | `{rows: [[time, label]], truncated}`, at most 10000 rows                                            |

`name` is the project name: the uploaded file's sanitised stem, or the YouTube video ID.
`track` is one of the CSVs in [output.md](output.md). Labels come decoded from `labels.json`.
Starting a name that is already queued or running returns that job. A project made in the
web UI is `done` as soon as its ZIP exists.

## How it runs

The `mcp` service runs the same image as the web app, with `uploads/` mounted read-only. A
job posts the audio to the web app's own endpoint, so every analysis takes the same path as
an upload in the browser ([ADR 0012](adr/0012-mcp-as-a-sibling-proxy.md)). Two jobs run at a
time; more than eight pending are refused. Jobs live in memory: after a restart, a finished
project still reads as `done` from disk, and one that was in flight reads as `unknown`.

## Security

Without `MCP_TOKEN` the service is loopback only and checks `Host` and `Origin` against DNS
rebinding. With `MCP_TOKEN` set, every request needs `Authorization: Bearer <token>`. With
`BIND` set to anything but a loopback address and no token, the service refuses to start.
See [security.md](security.md).

## Check it

`task dev:mcp:e2e` runs a fixture through a running service; `task docker:e2e` includes it.
````

- [ ] **Step 2: Write ADR 0012**

`docs/adr/0012-mcp-as-a-sibling-proxy.md`:

```markdown
# 0012. MCP as a sibling service that proxies to the web app

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

Agents should reach the service over MCP (Streamable HTTP). The MCP Python SDK serves ASGI;
the web app is Flask under gunicorn (WSGI). Analysis takes minutes and is serialised per
project by a `threading.Lock` in `pipeline.py`, which only holds inside one process.

## Decision

A second compose service, `mcp`, runs the same image under uvicorn. It mounts `uploads/`
read-only and starts each analysis by posting to the web app's own `/upload` or
`/download_yt`, in a background job the agent polls. Flask stays the only writer.

The MCP service takes an optional bearer token, `MCP_TOKEN`, and refuses to start without one
on a non-loopback `BIND`. This departs from ADR 0008 for this service only: 0008 rejected a
token because the browser UI could not send one, and an agent can.

## Consequences

**Gain.** No change to gunicorn, the lock or the web app's endpoints. The read-only mount
makes a second writer impossible, not just unlikely.

**Cost.** An extra HTTP hop and a second copy of each upload in memory. Jobs live in the MCP
process: a restart forgets a job in flight (it reads as `unknown`; its project finishes in
the web app regardless).

## Alternatives considered

**The MCP process calls `pipeline.build` itself, with a cross-process file lock.** No hop,
but the core lock changes, and `fcntl` does not exist on the Windows hosts the tests run on.

**One ASGI process serving both, Flask through a WSGI adapter.** One port and one lock
domain, but it replaces gunicorn and its timeout for the whole service.
```

- [ ] **Step 3: Update the indexes and pages**

`docs/adr/README.md`, after the 0011 row:

```markdown
| [0012](0012-mcp-as-a-sibling-proxy.md) | MCP as a sibling service that proxies to the web app | Accepted |
```

`docs/README.md`, after the `api.md` row:

```markdown
| [mcp.md](mcp.md) | MCP tools for agents, connecting, the token |
```

`app/README.md`, after the `youtube.py` row:

```markdown
| `mcp_server.py` | The MCP service: tool registration, Streamable HTTP at `/mcp`, token and Host checks ([docs/mcp.md](../docs/mcp.md)) |
| `mcp_tools.py` | The MCP tools and their argument checks; reads projects, never writes them |
| `jobs.py` | MCP analysis jobs: one per project name, two at a time, at most eight pending |
| `backend.py` | The MCP service's client for the web app's `/upload` and `/download_yt` |
```

`docs/configuration.md`, after the `HOST` row:

```markdown
| `MCP_PORT` | `8001` | compose, `mcp_server.py` | Port of the MCP endpoint. The Host check expects it |
| `MCP_TOKEN` | empty | `mcp_server.py` | Bearer token for the MCP endpoint. Required when `BIND` is not loopback |
| `BACKEND_URL` | `http://backend:8000` (compose) | `mcp_tools.py` | Where MCP jobs send audio |
```

`docs/security.md`, at the end of the "What is protected" table:

```markdown
| DNS rebinding against the MCP endpoint | Without a token: `Host` and `Origin` must be `127.0.0.1` or `localhost` on `MCP_PORT` | `create_app` in `app/mcp_server.py` |
| Unauthenticated MCP access off loopback | `MCP_TOKEN` bearer token, constant-time compare; no start without it on a public `BIND` | `app/mcp_server.py` |
| A second writer racing the project lock | The MCP service mounts `uploads/` read-only and analyses through the web app | `docker-compose.yml`, [ADR 0012](adr/0012-mcp-as-a-sibling-proxy.md) |
```

and in "Before setting `BIND=0.0.0.0`", a paragraph:

```markdown
The MCP endpoint is published on the same `BIND`. It refuses to start there without
`MCP_TOKEN`; set a long random one (`openssl rand -hex 32`).
```

`docs/architecture.md`, a section at the end:

```markdown
## The MCP service

A second compose service runs the same image under uvicorn and serves MCP at `/mcp`
([mcp.md](mcp.md)). It reads `uploads/` read-only. An analysis tool queues a job (`jobs.py`)
that posts the audio to the web app (`backend.py`), so every project is still built by
`pipeline.build` in the web app's process ([ADR 0012](adr/0012-mcp-as-a-sibling-proxy.md)).
```

`README.md`: where `docs/api.md` is linked, add the sentence
`Agents can connect over MCP: [docs/mcp.md](docs/mcp.md).`

- [ ] **Step 4: Verify the docs**

Run: `task gate T="quality:docs,lint:run"`
Expected: two `ok` lines.

- [ ] **Step 5: Commit**

```bash
git add docs/mcp.md docs/adr/0012-mcp-as-a-sibling-proxy.md docs/adr/README.md docs/README.md app/README.md docs/configuration.md docs/security.md docs/architecture.md README.md
git commit -m "docs(mcp): document the MCP service, its security and ADR 0012" -- docs/mcp.md docs/adr/0012-mcp-as-a-sibling-proxy.md docs/adr/README.md docs/README.md app/README.md docs/configuration.md docs/security.md docs/architecture.md README.md
```

---

### Task 8: Final gates

**Files:** none new.

- [ ] **Step 1: Full offline gate**

Run: `rtk task validate`
Expected: `validate: ok`.

- [ ] **Step 2: Dependency audit**

Run: `task gate T="security:deps"`
Expected: `ok    security:deps`. An advisory against `mcp`, `uvicorn` or a dependency they
bring in is reported to the person with its ID, not waived.

- [ ] **Step 3: Coverage of the new modules**

Run: `task gate T="coverage:run,coverage:check"`
Expected: two `ok` lines. A new module under its floor gets tests, not a baseline change.

- [ ] **Step 4: Receipt**

State which gates ran and their result, including `docker:e2e` from Task 6 Step 8.
