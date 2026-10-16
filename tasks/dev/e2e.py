#!/usr/bin/env python3
"""End-to-end check against a running server: upload a clip, verify the ZIP, check the
security refusals. Standard library only, so it runs against the container from any host.

Usage:
  python3 tasks/dev/e2e.py FILE [--url http://127.0.0.1:5000]

Uploads FILE (name it `fixture-*`), expects 200 and a ZIP holding audio.wav, meta.json and
the six track CSVs, then expects 403 for a cross-origin POST and 400 for a non-YouTube URL.
"""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

TRACKS = ("kick", "snare", "hh", "beats", "bass", "chords", "melody")


def request(url: str, data: bytes | None, headers: dict) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart(path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode()
    return head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode(), boundary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    failures = []

    body, boundary = multipart(args.file)
    started = time.monotonic()
    status, data = request(
        f"{base}/upload",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}", "Origin": base},
    )
    elapsed = time.monotonic() - started
    if status != 200:
        print(f"  FAIL  upload answered {status}: {data[:200]!r}")
        return 1
    name = args.file.stem
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    expected = {f"{name}/{t}.csv" for t in TRACKS} | {
        f"{name}/{f}" for f in ("audio.wav", "meta.json", "labels.json")
    }
    if missing := expected - names:
        failures.append(f"ZIP lacks {sorted(missing)}")
    print(f"  upload    200  {len(data)} bytes  {elapsed:.1f}s  {len(names)} files")

    status, _ = request(f"{base}/repair", b"", {"Origin": "https://evil.example"})
    print(f"  csrf      {status}")
    if status != 403:
        failures.append(f"cross-origin POST answered {status}, expected 403")

    payload = json.dumps({"url": "https://evil.example/?v=dQw4w9WgXcQ"}).encode()
    status, _ = request(
        f"{base}/download_yt", payload, {"Content-Type": "application/json", "Origin": base}
    )
    print(f"  yt-host   {status}")
    if status != 400:
        failures.append(f"foreign YouTube URL answered {status}, expected 400")

    for failure in failures:
        print(f"  FAIL  {failure}")
    print("e2e: ok" if not failures else f"e2e: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
