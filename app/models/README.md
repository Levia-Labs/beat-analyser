# app / models

| File | What it is |
| --- | --- |
| `nmp.onnx` | basic-pitch's ICASSP 2022 note model, ONNX export, from the `basic-pitch` 0.4.0 wheel on PyPI. sha256 `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec` |
| `LICENSE-basic-pitch` | Its licence, Apache 2.0, Copyright Spotify AB |

`task dev:basic-pitch` fetched both: it downloads the wheel, checks the sha256 PyPI
publishes for it, and extracts them. Run it again to refresh; `VERSION=` picks another
release. The code that runs the model is [notes.py](../notes.py), a port of basic-pitch's
inference whose header lists what changed from upstream.
