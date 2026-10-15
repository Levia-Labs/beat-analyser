# 0008. No authentication; loopback by default, same-origin POSTs

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

The service ran the Flask development server with the Werkzeug debugger on, bound to
`0.0.0.0:80`. Anyone reaching the port could execute code. It is a single-user tool;
accounts would be more machinery than the rest of it combined.

## Decision

No authentication. Instead: the container publishes on `127.0.0.1` unless `BIND` says
otherwise; the debugger is never enabled; POSTs from another origin are refused, because a
loopback-only server is still reachable from any web page the user opens.

## Consequences

**Cost.** Exposing it to a network needs a reverse proxy with authentication in front.
[docs/security.md](../security.md) says so.

**Benefit.** Safe defaults with no configuration, and nothing to log into.

## Alternatives considered

**A shared token in an environment variable.** Simple, and the browser UI would need a way
to send it. Worth doing if the service is ever shared; listed in TODO.md.

**CSRF tokens.** The Origin and `Sec-Fetch-Site` check covers the same attack for a
same-origin JavaScript client without server-side session state.
