# app

The service, and the build context of the Docker image. Modules import each other by bare
name; gunicorn loads the Flask object `app` from `app.py` in this directory.

| File | Role |
| --- | --- |
| `app.py` | Flask routes, request validation, same-origin check on POST, security headers |
| `pipeline.py` | One project from audio to ZIP, with a lock per project name; `repair_all` |
| `analysis.py` | madmom onset and note detection, band classification, CSV writing, ffmpeg decode |
| `storage.py` | Project names (the path-traversal boundary), `meta.json`, ZIP packaging, listing |
| `youtube.py` | URL to video ID, and the yt-dlp download |
| `cli.py` | The same pipeline from a terminal: analyse, `--youtube`, `--summary`, `--list`, `--repair` |
| `templates/index.html` | The page. No inline script or style (CSP) |
| `static/app.js`, `static/app.css` | Its behaviour and look |
| `requirements.txt` | Pinned runtime dependencies |
| `Dockerfile` | Two-stage build, non-root runtime, gunicorn |

How they fit together: [docs/architecture.md](../docs/architecture.md).
