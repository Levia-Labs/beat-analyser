# tests

pytest suite for the service and the repository tooling. `conftest.py` stubs madmom out and
replaces `analysis.analyse` with a stub that writes one CSV, so the HTTP and storage behaviour
runs without beatcore or the note model. Run all with `task test:run`, a subset with
`task test:run -- -k <expr>`. Property tests (`test_properties.py`) run 50 examples there and
1000 under `task props:deep`.

| File                    | What it covers                                                         |
| ----------------------- | ---------------------------------------------------------------------- |
| `conftest.py`           | Shared fixtures, Hypothesis profiles and the madmom stub               |
| `test_agent_tooling.py` | Agent-session tools: gate and outline                                  |
| `test_analysis.py`      | CSV track writing, label routing, end-to-end analysis                  |
| `test_app.py`           | Flask routes: CSP, upload, refusals, folders, download, repair         |
| `test_coverage_gate.py` | The coverage ratchet                                                   |
| `test_key.py`           | Key detection from pitch classes                                       |
| `test_notes.py`         | basic-pitch port; the model test is skipped without onnxruntime/ffmpeg |
| `test_properties.py`    | Properties of the trust-boundary functions and the CSV writer          |
| `test_repo_files.py`    | Guards on tracked text-file encodings                                  |
| `test_storage.py`       | Storage and name validation                                            |
| `test_youtube.py`       | YouTube URL reduction to video IDs                                     |
