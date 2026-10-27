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

Request bodies are capped at `MAX_UPLOAD_MB` x 1.5 + 1 MB, so a base64 upload of up to
`MAX_UPLOAD_MB` fits.

## How it runs

The `mcp` service runs the same image as the web app, with `uploads/` mounted read-only. A
job posts the audio to the web app's own endpoint, so every analysis takes the same path as
an upload in the browser ([ADR 0012](adr/0012-mcp-as-a-sibling-proxy.md)). Two jobs run at a
time; more than eight pending are refused. Jobs live in memory: after a restart, a finished
project still reads as `done` from disk, and one that was in flight reads as `unknown`. A job that failed stays `failed` in `project_status` even if the project
is later built another way, for example in the web UI; starting the analysis again clears it.

## Security

Without `MCP_TOKEN` the service is loopback only. It accepts `Host` and `Origin` values of
`127.0.0.1:<MCP_PORT>` and `localhost:<MCP_PORT>` only, which blocks DNS rebinding. With
`MCP_TOKEN` set, every request needs `Authorization: Bearer <token>`, and that token is the
check: the Host and Origin allowlist is off. With `BIND` set to anything but a loopback
address and no token, the service refuses to start. See [security.md](security.md).

## Check it

`task dev:mcp:e2e` runs a fixture through a running service; `task docker:e2e` includes it.
