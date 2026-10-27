# 0012. MCP as a sibling service that proxies to the web app

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

Agents should reach the service over MCP (Streamable HTTP). The MCP Python SDK serves ASGI;
the web app is Flask under gunicorn (WSGI). Analysis takes minutes and is serialised per
project by a `threading.Lock` in `pipeline.py`, which only holds inside one process.

## Decision

A second compose service, `mcp`, runs the same image under uvicorn. It mounts `uploads/`
read-only and starts each analysis by posting to the web app's own `/upload` or
`/download_yt`, in a background job the agent polls. Flask stays the only writer.

The MCP service takes an optional bearer token, `MCP_TOKEN`, and refuses to start without one
on a non-loopback `BIND`. This departs from ADR 0008 for this service only: 0008 rejected a
token because the browser UI could not send one, and an agent can.

## Consequences

**Gain.** No change to gunicorn, the lock or the web app's endpoints. The read-only mount
makes a second writer impossible, not just unlikely.

**Cost.** An extra HTTP hop and a second copy of each upload in memory. Jobs live in the MCP
process: a restart forgets a job in flight (it reads as `unknown`; its project finishes in
the web app regardless).

## Alternatives considered

**The MCP process calls `pipeline.build` itself, with a cross-process file lock.** No hop,
but the core lock changes, and `fcntl` does not exist on the Windows hosts the tests run on.

**One ASGI process serving both, Flask through a WSGI adapter.** One port and one lock
domain, but it replaces gunicorn and its timeout for the whole service.
