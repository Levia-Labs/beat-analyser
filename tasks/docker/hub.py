#!/usr/bin/env python3
"""The Docker Hub page of levialabs/beat-analyser: compare it with tasks/docker/hub.md, or
publish that file as the page. Standard library only.

Reading is public. Publishing signs in as DOCKERHUB_USERNAME with DOCKERHUB_TOKEN when both
are set, otherwise with what `docker login` saved, the same credentials `docker push` uses.
The secret must be a password or a personal access token with Read, Write, Delete scope;
Docker Hub refuses description edits with less.

Usage:
  python3 tasks/docker/hub.py            show what would change, change nothing
  python3 tasks/docker/hub.py --apply    publish the short and full description
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "levialabs/beat-analyser"
API = "https://hub.docker.com/v2"
# The key `docker login` files Docker Hub credentials under.
HUB_SERVER = "https://index.docker.io/v1/"
PAGE = Path(__file__).with_name("hub.md")
# Docker Hub caps the short description at 100 characters and the full one at 25,000.
SHORT = "Audio file or YouTube URL to timed drum, beat and note events as CSVs. Rust, ONNX, Flask."
SHORT_LIMIT, FULL_LIMIT = 100, 25_000


def call(method: str, url: str, body: dict | None = None, token: str = "") -> dict:
    headers = {"User-Agent": "beat-analyser-dev", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def docker_login() -> tuple[str, str]:
    """The Docker Hub username and secret `docker login` saved, the ones `docker push` uses.

    Either in a credential helper named by credsStore (Docker Desktop: `desktop`, which keeps
    them in the OS keychain), or base64 `user:secret` under auths in config.json.
    """
    config_dir = Path(os.environ.get("DOCKER_CONFIG") or Path.home() / ".docker")
    try:
        config = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    if store := config.get("credsStore"):
        try:
            found = subprocess.run(
                [f"docker-credential-{store}", "get"],
                input=HUB_SERVER,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            saved = json.loads(found.stdout)
            return saved.get("Username", ""), saved.get("Secret", "")
        except (OSError, subprocess.SubprocessError, ValueError):
            return "", ""
    encoded = config.get("auths", {}).get(HUB_SERVER, {}).get("auth", "")
    user, _, secret = base64.b64decode(encoded).decode().partition(":")
    return user, secret


def login() -> str:
    user, secret = os.environ.get("DOCKERHUB_USERNAME"), os.environ.get("DOCKERHUB_TOKEN")
    if not user or not secret:
        user, secret = docker_login()
    if not user or not secret:
        raise SystemExit("run `docker login`, or set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN")
    try:
        return call("POST", f"{API}/users/login", {"username": user, "password": secret})["token"]
    except urllib.error.HTTPError as exc:
        # A Docker Desktop browser sign-in saves a session token, not a password or access
        # token, and this endpoint refuses it.
        raise SystemExit(
            f"Docker Hub refused the login for {user} ({exc.code}). Run "
            "`docker login -u <user>` with a personal access token (Read, Write, Delete), "
            "or set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--apply", action="store_true", help="publish instead of comparing")
    args = parser.parse_args()

    full = PAGE.read_text(encoding="utf-8")
    if len(SHORT) > SHORT_LIMIT or len(full) > FULL_LIMIT:
        print(f"too long: short {len(SHORT)}/{SHORT_LIMIT}, full {len(full)}/{FULL_LIMIT}")
        return 1

    url = f"{API}/repositories/{REPOSITORY}/"
    try:
        current = call("GET", url)
        same_short = current.get("description") == SHORT
        same_full = current.get("full_description") == full
        print(f"short  {'unchanged' if same_short else 'differs'}  ({len(SHORT)} chars)")
        print(f"full   {'unchanged' if same_full else 'differs'}  ({len(full)} chars, {PAGE})")
        if not args.apply or (same_short and same_full):
            return 0
        call("PATCH", url, {"description": SHORT, "full_description": full}, token=login())
    except urllib.error.HTTPError as exc:
        print(f"{exc.code} {exc.reason}: {exc.read()[:300]!r}")
        return 1
    print(f"published https://hub.docker.com/r/{REPOSITORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
