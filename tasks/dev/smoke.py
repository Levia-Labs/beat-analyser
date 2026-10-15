#!/usr/bin/env python3
"""Probe a running server: the index loads with its security headers and /folders answers
JSON. Reads only; uploads nothing.

Usage:
  python3 tasks/dev/smoke.py [BASE_URL]     default http://127.0.0.1:5000
"""

import json
import sys
import urllib.error
import urllib.request

HEADERS = ("Content-Security-Policy", "X-Content-Type-Options", "X-Frame-Options")


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.headers, response.read()


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000").rstrip("/")
    failures = []
    try:
        status, headers, _ = get(f"{base}/")
        if status != 200:
            failures.append(f"/ answered {status}")
        failures += [f"/ lacks {name}" for name in HEADERS if name not in headers]

        status, _, body = get(f"{base}/folders")
        projects = json.loads(body)
        print(f"  /folders  {status}  {len(projects)} project(s)")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"{base}: {exc}. Is the server running? task run, or task docker:up")
        return 1

    for failure in failures:
        print(f"  FAIL  {failure}")
    print("smoke: ok" if not failures else f"smoke: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
