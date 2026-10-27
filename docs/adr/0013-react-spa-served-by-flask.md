# 0013. React SPA served by Flask from a Node build stage

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The page was one Jinja template with Bootstrap from a CDN and a script that fetched a ZIP.
The new UI needs a track viewer and the saber-tooth look, which a component framework and a
build step serve better. The service runs as one container behind a same-origin POST check
and a strict CSP.

## Decision

The SPA lives in `web/` (React, Vite, TypeScript, plain CSS). A Node stage in `app/Dockerfile`
builds it; Flask serves `index.html` for its routes and the hashed assets under `/assets/`.
Node never runs at runtime. The CDN is gone, so the CSP allows `'self'` only.

## Consequences

**Gain.** One origin, so the same-origin check and the CSP stay simple. One container and one
port (8002).

**Cost.** The image build needs Node and npm; a checkout without Node serves a 503 at `/`
until `task web:build`, while the API keeps working.

**Constraint.** Analysis progress (`app/progress.py`) lives in the process's memory, and the
SPA polls it from another request. That holds because the image runs one gunicorn worker
(several threads). A second worker would answer some polls without the entry; moving to
several workers means moving the registry out of process first.

## Alternatives considered

**An nginx container serving `dist/`.** Splits static files from the API, but adds a service,
a proxy configuration and a second origin to secure.
