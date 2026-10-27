# Configuration

Environment variables. docker-compose passes the ones marked compose through from the host
shell or a `.env` file next to `docker-compose.yml`.

| Variable         | Default                                 | Read by                  | Meaning                                                                             |
| ---------------- | --------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| `BIND`           | `127.0.0.1`                             | compose                  | Host address the port is published on. `0.0.0.0` exposes the service to the network |
| `PORT`           | `8002` (compose), `5000` (`task run`)   | compose, `app.py`        | Port                                                                                |
| `WEB_DIST`       | `web/dist` (`/app/web` in the image)    | app                      | Folder holding the built web app: `index.html` and its assets                       |
| `MAX_UPLOAD_MB`  | `200`                                   | app                      | Largest upload, and largest YouTube audio download                                  |
| `YT_MAX_MINUTES` | `20`                                    | app                      | Longest YouTube video accepted                                                      |
| `UPLOAD_ROOT`    | `uploads` (`/app/uploads` in the image) | app                      | Where projects live                                                                 |
| `HOST`           | `127.0.0.1`                             | `app.py` dev server      | Listen address for `task run`                                                       |
| `MCP_PORT`       | `8001`                                  | compose, `mcp_server.py` | Port of the MCP endpoint. Without `MCP_TOKEN` the Host check expects it             |
| `MCP_TOKEN`      | empty                                   | `mcp_server.py`          | Bearer token for the MCP endpoint. Required when `BIND` is not loopback             |
| `BACKEND_URL`    | `http://backend:8000` (compose)         | `mcp_tools.py`           | Where MCP jobs send audio                                                           |

The CLI takes `--root` instead of `UPLOAD_ROOT`; every root task passes `ROOT=` through
(`task analyse FILE=x.wav ROOT=/tmp/out`).

Gunicorn settings (workers, threads, timeout) are in the `CMD` of
[app/Dockerfile](../app/Dockerfile); see [architecture.md](architecture.md#runtime) for why.
