#!/usr/bin/env python3
"""The usage ceiling a long grind runs under, and the timer that resumes it.

A session that runs lanes until the allowance is gone loses whatever the last lane was in the
middle of, and leaves no way to tell a stall from a wall. This script is the gate: it reads the
same live windows the CLI's own `/usage` screen reads, compares them against a ceiling, and
answers in an exit code a shell or a lane can act on without a model in the loop.

    0   under the ceiling - keep working
    3   at or over it - retire the lanes and wait
    1   could not be determined (no credentials, endpoint moved, network down)

`--wait` is the other half: it blocks until every window is back under the ceiling, which is the
timer that lets an interrupted plan resume by itself rather than by somebody remembering. Waiting
for a *fresh* five-hour window is that wait aimed at one window with a low ceiling - see `--window`,
without which the seven-day figure, which does not return to zero on the hour, would hold the wait
open for days.

The endpoint is undocumented - the same one tasks/claude/Taskfile.yml calls, with the same caveat:
treat a shape change or a 404 as expected breakage and fall back to `claude` then `/usage`. The
access token is read from the credentials file and never printed.

Usage:
  python3 tasks/claude/budget.py                      # status, exit 0/3
  python3 tasks/claude/budget.py --ceiling 90         # a tighter ceiling
  python3 tasks/claude/budget.py --json               # machine readable
  python3 tasks/claude/budget.py --quiet              # exit code only
  python3 tasks/claude/budget.py --wait               # block until under the ceiling
  python3 tasks/claude/budget.py --wait --timeout 900 # give up waiting after 15 minutes
  python3 tasks/claude/budget.py --wait --window five_hour --ceiling 20
                                                     # block until the session window is fresh
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

CREDENTIALS = Path.home() / ".claude" / ".credentials.json"
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"

# Every window the endpoint reports, in the order a reader cares about them. A window absent from
# the payload is skipped rather than treated as zero: a missing number is not a spent one.
WINDOWS = [
    ("five_hour", "Session (5 hour)"),
    ("seven_day", "Weekly (7 day)"),
    ("seven_day_opus", "Weekly Opus"),
    ("seven_day_sonnet", "Weekly Sonnet"),
]

# How long each window is, which is what turns a single reading into a burn rate: a window that
# resets at a known time and runs a known length says how far into itself it already is.
WINDOW_MINUTES = {
    "five_hour": 5 * 60,
    "seven_day": 7 * 24 * 60,
    "seven_day_opus": 7 * 24 * 60,
    "seven_day_sonnet": 7 * 24 * 60,
}

DEFAULT_CEILING = 95.0
POLL_SECONDS = 120
MAX_NAP_SECONDS = 600
# The endpoint answers 429 to a tight poll, so no path may check faster than this.
MIN_POLL_SECONDS = 60

UNDER, ERROR, OVER = 0, 1, 3


def token():
    """The OAuth access token this machine is logged in with, or None under API-key auth."""
    if not CREDENTIALS.is_file():
        return None
    try:
        data = json.loads(CREDENTIALS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("claudeAiOauth", {}).get("accessToken") or None


def resets_at(window):
    """`resets_at` as an aware datetime.

    The payload carries a fractional part and a `+00:00` offset; older Pythons take neither, so the
    fraction is cut and a bare `Z` is read as UTC.
    """
    raw = window.get("resets_at")
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        offset = ""
        for marker in ("+", "-"):
            index = tail.find(marker)
            if index > 0:
                offset = tail[index:]
                break
        text = head + offset
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def fetch():
    """The usage payload, or None and the reason it could not be read."""
    access = token()
    if access is None:
        return (
            None,
            "No OAuth access token in the credentials file (API-key auth, or not logged in).",
        )

    request = urllib.request.Request(
        USAGE_URL,
        headers={"Authorization": "Bearer " + access, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as error:
        if error.code == 401:
            return (
                None,
                "401 - the stored access token has expired. Run any `claude` command, then retry.",
            )
        if error.code == 429:
            # Being rate-limited is not being out of allowance, and saying the endpoint moved would
            # send a reader to fix the wrong thing. Wait out the limit and ask again.
            return (
                None,
                "429 - the usage endpoint is rate-limiting this machine. "
                "Wait a minute and retry; this is not an exhausted allowance.",
            )
        return (
            None,
            f"HTTP {error.code} from the usage endpoint; it is undocumented and may have moved.",
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        return None, f"Could not read the usage endpoint: {error}"


def readings(payload, only=None):
    """One row per window the payload actually reports, or only the ones named.

    Naming a window is what makes waiting for a fresh session possible: the five-hour window returns
    to zero on the hour it says, while the seven-day one does not, so a wait judged against every
    window at a low ceiling would block for days rather than hours.
    """
    rows = []
    now = datetime.now(UTC)
    for key, label in WINDOWS:
        if only and key not in only:
            continue
        window = payload.get(key)
        if not isinstance(window, dict):
            continue
        used = window.get("utilization")
        if not isinstance(used, (int, float)):
            continue
        reset = resets_at(window)
        rows.append(
            {
                "window": key,
                "label": label,
                "utilization": float(used),
                "resets_at": reset.isoformat() if reset else None,
                "seconds_until_reset": max(0, int((reset - now).total_seconds()))
                if reset
                else None,
            }
        )
    return rows


def span(seconds):
    """A duration a reader can act on, rather than a count of seconds."""
    if seconds is None:
        return "unknown"
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def report(rows, ceiling):
    for row in rows:
        filled = min(20, int(row["utilization"] / 5))
        bar = "#" * filled + "." * (20 - filled)
        mark = "OVER" if row["utilization"] >= ceiling else "ok"
        print(
            "{:<18} [{}] {:5.1f}%  resets in {:<8} {}".format(
                row["label"], bar, row["utilization"], span(row["seconds_until_reset"]), mark
            )
        )


def breached(rows, ceiling):
    return [row for row in rows if row["utilization"] >= ceiling]


def soonest_reset(rows):
    """How long until the first breached window comes back, in seconds."""
    waits = [row["seconds_until_reset"] for row in rows if row["seconds_until_reset"] is not None]
    return min(waits) if waits else POLL_SECONDS


def check(args):
    payload, reason = fetch()
    if payload is None:
        if args.json:
            print(json.dumps({"ok": False, "reason": reason}, indent=2))
        elif not args.quiet:
            print(reason, file=sys.stderr)
        return ERROR

    rows = readings(payload, args.window)
    if not rows:
        if not args.quiet:
            print("The usage endpoint reported no windows; its shape has changed.", file=sys.stderr)
        return ERROR

    over = breached(rows, args.ceiling)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "ceiling": args.ceiling,
                    "over_ceiling": bool(over),
                    "breached": [row["window"] for row in over],
                    "wait_seconds": soonest_reset(over) if over else 0,
                    "windows": rows,
                },
                indent=2,
            )
        )
        return OVER if over else UNDER

    if not args.quiet:
        report(rows, args.ceiling)
        if over:
            names = ", ".join(row["label"] for row in over)
            print("")
            print(f"At or over the {args.ceiling:g}% ceiling: {names}.")
            print(f"Retire the lanes; the first window returns in {span(soonest_reset(over))}.")
    return OVER if over else UNDER


def wait(args):
    """Block until every window is under the ceiling. The timer half of the gate."""
    deadline = time.monotonic() + args.timeout if args.timeout else None
    while True:
        payload, reason = fetch()
        if payload is None:
            # A network blip must not look like an exhausted allowance: say so and keep waiting.
            nap = POLL_SECONDS
            if not args.quiet:
                print(f"waiting - {reason}", file=sys.stderr, flush=True)
        else:
            rows = readings(payload, args.window)
            over = breached(rows, args.ceiling)
            if not over:
                if not args.quiet:
                    report(rows, args.ceiling)
                    print("")
                    print(f"Under the {args.ceiling:g}% ceiling. Resume.")
                return UNDER
            nap = min(soonest_reset(over) + 30, MAX_NAP_SECONDS)
            if not args.quiet:
                names = ", ".join(row["label"] for row in over)
                print(
                    f"{names} over {args.ceiling:g}% - next check in {span(nap)}.",
                    flush=True,
                )

        if deadline is not None:
            left = deadline - time.monotonic()
            # A deadline shortens the nap but never below the floor: the endpoint answers 429 to a
            # tight poll, and a run that turns its own wait into a rate limit reports the wrong
            # reason for stopping. Too little time left to poll again is the deadline arriving.
            if left <= MIN_POLL_SECONDS:
                if left > 0:
                    time.sleep(left)
                if not args.quiet:
                    print("Timed out still over the ceiling.", file=sys.stderr)
                return OVER
            nap = min(nap, int(left))
        time.sleep(max(MIN_POLL_SECONDS, nap))


def main():
    parser = argparse.ArgumentParser(
        description="The usage ceiling a long grind runs under, and the timer that resumes it.",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=DEFAULT_CEILING,
        help="percent utilisation that retires a lane (default 95)",
    )
    parser.add_argument(
        "--window",
        action="append",
        choices=[key for key, _ in WINDOWS],
        help="judge only this window (repeatable); default is every window reported",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable status")
    parser.add_argument("--quiet", action="store_true", help="exit code only")
    parser.add_argument("--wait", action="store_true", help="block until under the ceiling")
    parser.add_argument(
        "--timeout", type=int, default=0, help="seconds to wait before giving up (0 = forever)"
    )
    args = parser.parse_args()
    return wait(args) if args.wait else check(args)


if __name__ == "__main__":
    sys.exit(main())
