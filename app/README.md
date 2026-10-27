# app

The service, and the build context of the Docker image. Modules import each other by bare
name; gunicorn loads the Flask object `app` from `app.py` in this directory.

| File               | Role                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| `app.py`           | Flask routes, request validation, same-origin check on POST, security headers                                   |
| `pipeline.py`      | One project from audio to ZIP, with a lock per project name; `repair_all`                                       |
| `progress.py`      | Analysis stages the SPA polls; in memory, one worker                                                            |
| `analysis.py`      | Calls beatcore (rhythm) and `notes` (pitch), writes the track CSVs, ffmpeg decode                               |
| `jobs.py`          | MCP analysis jobs: one per project name, two at a time, at most eight pending                                   |
| `backend.py`       | Client for the web app's `/upload` and `/download_yt`, used by the MCP jobs                                     |
| `mcp_server.py`    | The MCP service: tool registration, Streamable HTTP at /mcp, token and Host checks                              |
| `events.py`        | Reads track CSVs, labels.json and meta.json; shared by the web app and the MCP tools                            |
| `mcp_tools.py`     | The MCP tools and their argument checks; reads projects, never writes them                                      |
| `notes.py`         | basic-pitch note transcription on onnxruntime - a port, changes listed in its header                            |
| `key.py`           | Musical key from the notes: Krumhansl-Schmuckler profiles, `unknown` when nothing fits                          |
| `models/`          | The basic-pitch ONNX model and its licence ([models/README.md](models/README.md))                               |
| `storage.py`       | Project names (the path-traversal boundary), `meta.json`, ZIP packaging, listing                                |
| `youtube.py`       | URL to video ID, and the yt-dlp download                                                                        |
| `cli.py`           | The same pipeline from a terminal: analyse, `--youtube`, `--summary`, `--list`, `--repair`                      |
| `requirements.txt` | Pinned runtime dependencies                                                                                     |
| `Dockerfile`       | Three-stage build (Rust toolchain, wheel and venv, runtime), non-root, gunicorn. Built from the repository root |

The rhythm engine is the Rust crate in [../beatcore/](../beatcore/README.md). How everything
fits together: [docs/architecture.md](../docs/architecture.md).
