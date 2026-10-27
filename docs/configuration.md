# Configuration

Environment variables. docker-compose passes the ones marked compose through from the host
shell or a `.env` file next to `docker-compose.yml`.

| Variable            | Default                                 | Read by                    | Meaning                                                                                                                                                                                      |
| ------------------- | --------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BIND`              | `127.0.0.1`                             | compose, app               | Host address the port is published on. `0.0.0.0` exposes the service to the network; the app then refuses to start without `WEB_TOKEN`                                                       |
| `PORT`              | `8002` (compose), `5000` (`task run`)   | compose, `app.py`          | Port                                                                                                                                                                                         |
| `WEB_DIST`          | `web/dist` (`/app/web` in the image)    | app                        | Folder holding the built web app: `index.html` and its assets                                                                                                                                |
| `MAX_UPLOAD_MB`     | `200`                                   | app                        | Largest upload, and largest YouTube audio download                                                                                                                                           |
| `MAX_ANALYSES`      | `2`                                     | app                        | Analyses running at once; a request that would start another gets 429                                                                                                                        |
| `YT_MAX_MINUTES`    | `20`                                    | app                        | Longest YouTube video accepted                                                                                                                                                               |
| `UPLOAD_ROOT`       | `uploads` (`/app/uploads` in the image) | app                        | Where projects live                                                                                                                                                                          |
| `HOST`              | `127.0.0.1`                             | `app.py` dev server        | Listen address for `task run`                                                                                                                                                                |
| `WEB_TOKEN`         | empty                                   | compose, app, `backend.py` | Token the API requires (Bearer header or session cookie), at least 16 characters or the app refuses to start. Required when `BIND` is not loopback; compose passes it to the MCP service too |
| `WEB_COOKIE_SECURE` | unset                                   | app                        | `1` makes the session cookie always `Secure`, for a TLS-terminating proxy; otherwise it is `Secure` only when the request itself is https                                                    |
| `MCP_PORT`          | `8001`                                  | compose, `mcp_server.py`   | Port of the MCP endpoint. Without `MCP_TOKEN` the Host check expects it                                                                                                                      |
| `MCP_TOKEN`         | empty                                   | `mcp_server.py`            | Bearer token for the MCP endpoint. Required when `BIND` is not loopback                                                                                                                      |
| `BACKEND_URL`       | `http://backend:8000` (compose)         | `mcp_tools.py`             | Where MCP jobs send audio                                                                                                                                                                    |

The CLI takes `--root` instead of `UPLOAD_ROOT`; every root task passes `ROOT=` through
(`task analyse FILE=x.wav ROOT=/tmp/out`).

Gunicorn settings (workers, threads, timeout) are in the `CMD` of
[app/Dockerfile](../app/Dockerfile); see [architecture.md](architecture.md#runtime) for why.
