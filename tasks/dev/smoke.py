#!/usr/bin/env python3
"""Probe a running server: the index loads with its security headers and /folders answers
JSON. Reads only; uploads nothing.

Usage:
  python3 tasks/dev/smoke.py [BASE_URL] [--token T]     default http://127.0.0.1:5000

--token, default the WEB_TOKEN environment variable, is sent as Authorization: Bearer.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HEADERS = ("Content-Security-Policy", "X-Content-Type-Options", "X-Frame-Options")


def get(url: str, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=10
    ) as response:
        return response.status, response.headers, response.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:5000")
    parser.add_argument("--token", default=os.environ.get("WEB_TOKEN"))
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")
    failures = []
    try:
        status, headers, _ = get(f"{base}/", args.token)
        if status != 200:
            failures.append(f"/ answered {status}")
        failures += [f"/ lacks {name}" for name in HEADERS if name not in headers]

        status, _, body = get(f"{base}/folders", args.token)
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
