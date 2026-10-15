#!/usr/bin/env python3
"""Fetch Spotify's basic-pitch model from its pinned PyPI wheel, verified by sha256.

app/notes.py is a slim port of basic-pitch's ONNX inference. This script is how its model
and licence got into the repository, and how they are refreshed: it downloads the wheel
named below, checks the digest PyPI publishes, and extracts

  basic_pitch/saved_models/icassp_2022/nmp.onnx  ->  app/models/nmp.onnx
  LICENSE                                         ->  app/models/LICENSE-basic-pitch
  the upstream inference sources                  ->  .cache/basic-pitch-src/ (reference)

Usage:
  python3 tasks/dev/basic_pitch.py [--version 0.4.0]
"""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "app" / "models"
REFERENCE = ROOT / ".cache" / "basic-pitch-src"
SOURCES = ("constants.py", "inference.py", "note_creation.py", "__init__.py")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "beat-analyser-dev"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="0.4.0")
    args = parser.parse_args()

    meta = json.loads(fetch(f"https://pypi.org/pypi/basic-pitch/{args.version}/json"))
    wheel = next(f for f in meta["urls"] if f["filename"].endswith(".whl"))
    data = fetch(wheel["url"])
    digest = hashlib.sha256(data).hexdigest()
    if digest != wheel["digests"]["sha256"]:
        print(f"sha256 mismatch for {wheel['filename']}: {digest}", file=sys.stderr)
        return 1
    print(f"{wheel['filename']}  sha256 {digest}  verified")

    MODELS.mkdir(parents=True, exist_ok=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        model = next(n for n in names if n.endswith("icassp_2022/nmp.onnx"))
        (MODELS / "nmp.onnx").write_bytes(archive.read(model))
        licence = next(n for n in names if n.endswith(("LICENSE", "LICENSE.txt")))
        (MODELS / "LICENSE-basic-pitch").write_bytes(archive.read(licence))
        for name in names:
            if name.startswith("basic_pitch/") and Path(name).name in SOURCES:
                (REFERENCE / Path(name).name).write_bytes(archive.read(name))
    model_digest = hashlib.sha256((MODELS / "nmp.onnx").read_bytes()).hexdigest()
    print(f"app/models/nmp.onnx  sha256 {model_digest}")
    print(f"reference sources in {REFERENCE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
