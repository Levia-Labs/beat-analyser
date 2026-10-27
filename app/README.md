# app

The service, and the build context of the Docker image. Modules import each other by bare
name; gunicorn loads the Flask object `app` from `app.py` in this directory.

| File                              | Role                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `app.py`                          | Flask routes, request validation, same-origin check on POST, security headers                                   |
| `pipeline.py`                     | One project from audio to ZIP, with a lock per project name; `repair_all`                                       |
| `analysis.py`                     | Calls beatcore (rhythm) and `notes` (pitch), writes the track CSVs, ffmpeg decode                               |
| `jobs.py`                         | in-memory registry of MCP analysis jobs, one per project name, run in a thread pool                             |
| `backend.py`                      | Client for the web app's `/upload` and `/download_yt`, used by the MCP jobs                                     |
| `mcp_server.py`                   | The MCP service: the tools over Streamable HTTP at /mcp, optional bearer token                                  |
| `mcp_tools.py`                    | The MCP tools: start analyses, poll them, read events; every argument checked before use                        |
| `notes.py`                        | basic-pitch note transcription on onnxruntime - a port, changes listed in its header                            |
| `key.py`                          | Musical key from the notes: Krumhansl-Schmuckler profiles, `unknown` when nothing fits                          |
| `models/`                         | The basic-pitch ONNX model and its licence ([models/README.md](models/README.md))                               |
| `storage.py`                      | Project names (the path-traversal boundary), `meta.json`, ZIP packaging, listing                                |
| `youtube.py`                      | URL to video ID, and the yt-dlp download                                                                        |
| `cli.py`                          | The same pipeline from a terminal: analyse, `--youtube`, `--summary`, `--list`, `--repair`                      |
| `templates/index.html`            | The page. No inline script or style (CSP)                                                                       |
| `static/app.js`, `static/app.css` | Its behaviour and look                                                                                          |
| `requirements.txt`                | Pinned runtime dependencies                                                                                     |
| `Dockerfile`                      | Three-stage build (Rust toolchain, wheel and venv, runtime), non-root, gunicorn. Built from the repository root |

The rhythm engine is the Rust crate in [../beatcore/](../beatcore/README.md). How everything
fits together: [docs/architecture.md](../docs/architecture.md).
