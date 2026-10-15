#!/usr/bin/env python3
"""Popularity and freshness of candidate libraries, from the public GitHub and PyPI APIs.
Standard library only; no token needed (GitHub allows 60 unauthenticated calls an hour).

Usage:
  python3 tasks/dev/packages.py --github spotify/basic-pitch librosa/librosa
  python3 tasks/dev/packages.py --pypi basic-pitch librosa onnxruntime
  python3 tasks/dev/packages.py --crates pyo3 realfft hound
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "beat-analyser-dev"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def github(repo: str) -> str:
    d = fetch(f"https://api.github.com/repos/{repo}")
    licence = (d.get("license") or {}).get("spdx_id")
    return (
        f"{d['full_name']:<45} {d['stargazers_count']:>7} stars  {licence or '-':<12} "
        f"pushed {d['pushed_at'][:10]}  {d.get('language') or '-'}"
        + ("  ARCHIVED" if d.get("archived") else "")
    )


def pypi(package: str) -> str:
    d = fetch(f"https://pypi.org/pypi/{package}/json")
    info, version = d["info"], d["info"]["version"]
    files = d["releases"].get(version) or [{}]
    tags = sorted({f["filename"].split("-")[-3] for f in files if f["filename"].endswith(".whl")})
    deps = [r for r in info.get("requires_dist") or [] if "extra ==" not in r]
    return (
        f"{info['name']} {version} ({files[0].get('upload_time', '')[:10]}) "
        f"python {info.get('requires_python') or '-'}  wheels {','.join(tags[-5:]) or 'sdist'}\n"
        f"    deps: {'; '.join(deps[:10]) or '-'}"
    )


def crate(name: str) -> str:
    d = fetch(f"https://crates.io/api/v1/crates/{name}")["crate"]
    return (
        f"{d['name']:<20} {d['max_stable_version']:<10} updated {d['updated_at'][:10]}  "
        f"{d['downloads']:>12,} downloads"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--github", nargs="*", default=[], metavar="OWNER/REPO")
    parser.add_argument("--pypi", nargs="*", default=[], metavar="PACKAGE")
    parser.add_argument("--crates", nargs="*", default=[], metavar="CRATE")
    args = parser.parse_args()
    failed = 0
    for lookup, names in ((github, args.github), (pypi, args.pypi), (crate, args.crates)):
        for name in names:
            try:
                print(lookup(name))
            except (urllib.error.URLError, KeyError, ValueError) as exc:
                failed += 1
                print(f"{name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
