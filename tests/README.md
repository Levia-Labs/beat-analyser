# tests

pytest suite for the service and the repository tooling. `conftest.py` stubs madmom out and
replaces `analysis.analyse` with a stub that writes one CSV, so the HTTP and storage behaviour
runs without beatcore or the note model. Run all with `task test:run`, a subset with
`task test:run -- -k <expr>`. Property tests (`test_properties.py`) run 50 examples there and
1000 under `task props:deep`.

| File                    | What it covers                                                          |
| ----------------------- | ----------------------------------------------------------------------- |
| `conftest.py`           | Shared fixtures, Hypothesis profiles and the madmom stub                |
| `test_agent_tooling.py` | Agent-session tools: gate and outline                                   |
| `test_analysis.py`      | CSV track writing, label routing, end-to-end analysis                   |
| `test_api.py`           | JSON answers for the SPA: name instead of ZIP, config, projects, events |
| `test_app.py`           | Flask routes: CSP, upload, refusals, folders, download, repair          |
| `test_backend.py`       | The MCP web app client: multipart body, error messages, unreachable     |
| `test_coverage_gate.py` | The coverage ratchet                                                    |
| `test_commit.py`        | The path-scoped commit helper against a temporary git repo              |
| `test_pyapi.py`         | The Python API inspector: attribute, listing, refusals                  |
| `test_events.py`        | Shared event reading: lane order, malformed rows, project list, events  |
| `test_key.py`           | Key detection from pitch classes                                        |
| `test_jobs.py`          | the MCP job registry: dedupe per name, pending cap, error capture       |
| `test_mcp_e2e.py`       | The MCP end-to-end script: tool reply reading, polling to completion    |
| `test_mcp_server.py`    | The MCP server: tools served in-process, bearer token, start-up refusal |
| `test_mcp_tools.py`     | The MCP tools: argument refusals, job start, status, event windows      |
| `test_notes.py`         | basic-pitch port; the model test is skipped without onnxruntime/ffmpeg  |
| `test_progress.py`      | Progress registry, stage callbacks, the poll route                      |
| `test_properties.py`    | Properties of the trust-boundary functions and the CSV writer           |
| `test_repo_files.py`    | Guards on tracked text-file encodings                                   |
| `test_storage.py`       | Storage and name validation                                             |
| `test_youtube.py`       | YouTube URL reduction to video IDs                                      |
