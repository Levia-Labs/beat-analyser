# MCP server for agents

- **Date:** 2026-10-27
- **Status:** Approved design

## Goal

Let agents connect to the service over the Model Context Protocol and analyse audio, check
progress and read events without going through the web UI or the ZIP.

## Decisions (from the user)

1. Transport: Streamable HTTP, served from a container built from the service image.
2. Audio input: a YouTube URL, or base64 audio in the tool call. Line-wrapped base64 (the
   `base64` command wraps at 76 columns) is accepted.
3. Long analyses: a job the agent polls, not a blocking call.
4. Architecture: a sibling proxy service. Flask stays the only writer of `uploads/`.

## Facts the design rests on

1. The per-project lock is a `threading.Lock` (`app/pipeline.py:17-23`). It serialises work
   inside one process only, so a second process that called `pipeline.build` would race the
   web app on the same folder.
2. `same_origin_posts_only` (`app/app.py:71-81`) refuses a POST only on `Sec-Fetch-Site:
cross-site` or a foreign `Origin`. A server-to-server POST sends neither and passes.
3. Analysis is synchronous inside the request; gunicorn's worker timeout is 1800 s for that
   reason (`app/Dockerfile:61-63`).
4. `mcp` 2.3.0 and `uvicorn` 0.54.0 publish py3 wheels and need Python >= 3.10. The image runs
   Python 3.14; `task docker:e2e` is what proves the combination.

## Architecture

A second service in `docker-compose.yml`, `mcp`, runs the same image `beat-analyser:latest`
with a different command:

```yaml
mcp:
  image: beat-analyser:latest
  command: ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
  ports:
    - "${BIND:-127.0.0.1}:${MCP_PORT:-8001}:8001"
  volumes:
    - ./uploads:/app/uploads:ro
  environment:
    BACKEND_URL: http://backend:8000
    MCP_TOKEN: ${MCP_TOKEN:-}
    BIND: ${BIND:-127.0.0.1}
  depends_on:
    backend:
      condition: service_healthy
```

1. One uvicorn worker: the job registry lives in that process's memory.
2. `uploads/` is mounted read-only, so the MCP service cannot write a project even by mistake.
   Every analysis goes through Flask's existing endpoints and lock.
3. Agents connect to `http://127.0.0.1:8001/mcp`. Claude Code:
   `{"mcpServers": {"beat-analyser": {"type": "http", "url": "http://127.0.0.1:8001/mcp"}}}`.
   The snippet goes in the docs, not in a committed `.mcp.json`.

## Components

All four are new modules in `app/`, importing each other and `storage` and `youtube` by bare
name (convention 1). Each stays under 300 lines.

| Module          | Role                                                                                                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mcp_tools.py`  | The tool functions and the checks they make. Validates every argument before a job is created                                                                                             |
| `mcp_server.py` | The `MCPServer` (SDK 2.x; `FastMCP` in 1.x), tool registration, transport security, the token middleware, the ASGI `app`                                                                  |
| `jobs.py`       | Job registry keyed by project name: `queued`, `running`, `done`, `failed`, with the error text. A `ThreadPoolExecutor(max_workers=2)`; more than 8 pending jobs is refused                |
| `backend.py`    | HTTP client for `POST /upload` (multipart) and `POST /download_yt` (JSON) against `BACKEND_URL`, timeout 1800 s. A non-2xx answer becomes a `BackendError` carrying the backend's message |

`jobs.py` takes its executor and its backend call as parameters so tests run jobs inline
with a stub backend.

## Tools

| Tool              | Arguments                                            | Returns                                      |
| ----------------- | ---------------------------------------------------- | -------------------------------------------- |
| `analyse_youtube` | `url`                                                | `{name, status}` at once                     |
| `analyse_audio`   | `filename`, `data_base64`                            | `{name, status}` at once                     |
| `project_status`  | `name`                                               | `{name, status, error?, tracks?}`            |
| `list_projects`   | none                                                 | project names with their `meta.json` content |
| `get_events`      | `name`, `track`, `start=0`, `end=None`, `limit=1000` | `{rows: [[time, label], ...], truncated}`    |

1. `analyse_youtube` reduces the URL with `youtube.video_id` before anything else (convention
   2). The project name is the video ID, as the web endpoint makes it.
2. `analyse_audio` checks the extension against the same allowed set as `app.py`, derives the
   name with `storage.project_name`, and rejects a payload whose base64 length implies more
   than `MAX_CONTENT_LENGTH` bytes before decoding it. Invalid base64 is refused.
3. A second analysis of a name already `queued` or `running` returns the existing job rather
   than starting another.
4. `project_status` reads the registry first. A name the registry does not know is resolved
   from disk: a project with its ZIP is `done`, anything else is `unknown`. Only the ZIP
   counts: `pipeline.build` writes `meta.json` before packaging, so a folder with
   `meta.json` and no ZIP may still be building. This keeps finished work visible after the MCP service restarts. When `done`,
   `tracks` lists each track CSV with its event count and first and last time.
5. `list_projects` wraps `storage.list_projects` over the read-only root.
6. `get_events` checks `name` with `storage.is_valid_name` and accepts `track` only if it is
   one of the CSV stems present in that project folder. Rows are filtered to
   `start <= time < end`, capped at `limit` (at most 10000); `truncated` says whether rows
   were left out.

Tool errors (validation, refused job, unknown project) are returned as MCP tool errors with a
one-line message, the same wording as the HTTP API where one exists.

## Security

1. The port is published on `127.0.0.1` by default, as the web service is (ADR 0008).
2. Without a token, Host and Origin allowlists (`127.0.0.1:<MCP_PORT>`, `localhost:<MCP_PORT>`)
   from the SDK's transport security settings block DNS rebinding from a browser page. Agents
   send no `Origin`. With a token, the allowlist is off: a rebinding page cannot know the
   token, and the Host a remote agent sends is not known in advance.
3. When `MCP_TOKEN` is set, every request needs `Authorization: Bearer <token>`, compared with
   `hmac.compare_digest`. The service refuses to start without a token when `BIND` is not a
   loopback address. This departs from ADR 0008 for the MCP service only: an agent can send a
   header, which the browser UI could not, so the reason 0008 gave against a token does not
   apply. A new ADR records this and the sibling-proxy decision.
4. The read-only mount, name and URL validation, the size check and the job cap close the
   rest of the boundary.

## Error handling

1. Backend unreachable or timed out: the job is `failed` with that message.
2. Backend 4xx or 5xx: the job is `failed` with the backend's `error` field.
3. An exception inside a job thread is caught, logged and recorded as `failed`; it never
   kills the executor.
4. Malformed CSV rows in `get_events` are skipped, not raised.

## Testing

Tests first, in the host virtualenv, no network:

1. `tests/test_jobs.py`: state transitions, duplicate submission, the pending cap, failure
   recording, disk fallback after a restart.
2. `tests/test_backend.py`: request shape for both endpoints and error mapping, against a stub
   transport.
3. `tests/test_mcp_tools.py`: each tool called in-process with jobs run inline and the backend
   stubbed; validation refusals; `get_events` windowing and truncation on a temporary
   project.
4. `tests/test_properties.py`: a property for the `analyse_audio` input check (filename and
   base64 size), a trust boundary.
5. `tasks/dev/mcp_e2e.py` behind `task dev:mcp:e2e`: initialize, `analyse_audio` on a fixture,
   poll `project_status`, `get_events`, and a refused bad name. `task docker:e2e` starts the
   `mcp` service and calls it. The task gets a test in `tests/test_mcp_e2e.py`, a row in
   `tasks/README.md` and a line in `docs/tasks.md`.

`app/requirements.txt`, `app/Dockerfile` (if touched) and `docker-compose.yml` change, so
`task docker:e2e` is required, plus `task security:deps` for the new dependencies.

## Dependencies

`mcp` and `uvicorn` pinned in `app/requirements.txt`. The HTTP client in `backend.py` is the
standard library's `urllib`, as `tasks/dev/e2e.py` already uses: one blocking POST needs
nothing more, and no client library is added.

## Docs

1. `docs/mcp.md`: connecting, tools, polling, token. Linked from `docs/README.md` and the root
   README.
2. ADR 0012: sibling proxy over a second writer, and the token for the MCP service.
3. `app/README.md`: rows for the three modules.
4. `docs/configuration.md`: `MCP_PORT`, `MCP_TOKEN`, `BACKEND_URL`. The repository has no
   `.env.example`; the configuration page is where variables are documented.
5. `docs/security.md`: the MCP boundary.
6. `docs/architecture.md`: the second service.

## Out of scope

1. A stdio transport.
2. MCP resources and prompts: tools cover the use.
3. Cancelling a running job: the backend request cannot be interrupted.
4. Exposing `/repair` over MCP.
