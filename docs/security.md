# Security

The service has no authentication. It is meant for one person on one machine, and the
defaults keep it there: compose publishes the port on `127.0.0.1` only.

## Before setting `BIND=0.0.0.0`

Anyone who can reach the port can upload audio, make the server download from YouTube,
fill the disk and start `/repair`, which reanalyses every song. Put it behind a reverse
proxy with authentication and request-size limits, or keep it on loopback.

The MCP endpoint is published on the same `BIND`. It refuses to start there without
`MCP_TOKEN`; set a long random one (`openssl rand -hex 32`).

## What is protected

| Threat                                                       | Protection                                                                                                         | Where                                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Remote code execution through the Werkzeug debugger          | Debugger never enabled; gunicorn in the container                                                                  | `app/app.py`, `app/Dockerfile`                                        |
| Path traversal through an upload name or `/download/<name>`  | Names sanitised and matched against one pattern before any path is built                                           | `storage.project_name`, `storage.is_valid_name`                       |
| yt-dlp fetching from an arbitrary host                       | Client URL reduced to an 11-character video ID and rebuilt as a youtube.com URL                                    | `youtube.video_id`                                                    |
| Oversize uploads and downloads                               | `MAX_UPLOAD_MB` on both; `YT_MAX_MINUTES` on video length                                                          | `app/app.py`, `youtube.download_audio`                                |
| Cross-site request forgery (a web page writing to localhost) | Every method but GET, HEAD, OPTIONS with a foreign `Origin` or `Sec-Fetch-Site: cross-site` refused                | `same_origin_posts_only` in `app/app.py`                              |
| Stored XSS through project or file names                     | React escapes rendered text; same-origin-only CSP with no inline script or style                                   | `CSP` in `app/app.py`                                                 |
| Container breakout impact                                    | Non-root user, read-only root file system, all capabilities dropped, `no-new-privileges`                           | `app/Dockerfile`, `docker-compose.yml`                                |
| Concurrent writes to one project                             | Lock per project name                                                                                              | `app/pipeline.py`                                                     |
| Deleting or renaming the wrong or a busy project             | Both names pass `storage.is_valid_name`; the project lock is tried without waiting, so a running analysis gets 409 | `delete_project`, `rename_project` in `app/app.py`, `app/pipeline.py` |
| DNS rebinding against the MCP endpoint                       | Without a token: `Host` and `Origin` must be `127.0.0.1` or `localhost` on `MCP_PORT`                              | `create_app` in `app/mcp_server.py`                                   |
| Unauthenticated MCP access off loopback                      | `MCP_TOKEN` bearer token, constant-time compare; no start without it on a public `BIND`                            | `app/mcp_server.py`                                                   |
| A second writer racing the project lock                      | The MCP service mounts `uploads/` read-only and analyses through the web app                                       | `docker-compose.yml`, [ADR 0012](adr/0012-mcp-as-a-sibling-proxy.md)  |
| Oversize MCP requests                                        | Request bodies capped at `MAX_UPLOAD_MB` x 1.5 + 1 MB, so a base64 upload up to `MAX_UPLOAD_MB` fits               | `body_limit` in `app/mcp_server.py`                                   |

## What is not

1. No authentication or rate limiting.
2. `/repair` runs synchronously and can occupy a worker thread for a long time.
3. yt-dlp is unpinned by decision ([ADR 0004](adr/0004-yt-dlp-unpinned.md)); a new release
   is trusted on rebuild.
4. The same-origin check compares `Origin` with the `Host` the app sees. A reverse proxy
   that rewrites `Host` makes every browser POST fail with 403; forward the original host
   header (or add Werkzeug's `ProxyFix`).
5. Any local process, or anyone who reaches a non-loopback port, can delete or rename a
   project; the same-origin check only stops other web pages.

## Reporting

Security issues go to the maintainers privately, not to a public issue.
