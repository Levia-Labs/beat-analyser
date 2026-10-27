# 0015. An optional web token, held by the browser as a session cookie

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

[ADR 0008](0008-loopback-without-auth.md) chose no authentication and told anyone sharing the
service to put a reverse proxy in front. The MCP service already has an optional bearer
token and refuses a public `BIND` without it, so the web app was the one port that could be
published open. The SPA also has plain links for the ZIP downloads, and a link cannot carry
an `Authorization` header.

## Decision

`WEB_TOKEN`, when set, is required by every route except the SPA shell (`/`, `/add`,
`/p/<name>`, `/assets/*`), `GET /config` and the login routes. A request passes with
`Authorization: Bearer <token>` or the `ba_session` cookie. `POST /login` checks the token
in constant time and sets the cookie: `HttpOnly`, `SameSite=Strict`, path `/`, `Secure` over
https, no expiry; its value is an HMAC of a fixed label keyed by the token. A wrong token
waits one second before the 401. The app exits at import when `WEB_TOKEN` is unset and
`BIND` is not loopback. The MCP service sends the same token to the backend. Without
`WEB_TOKEN` nothing changes: this amends ADR 0008 only for when the token is set.

## Consequences

**Cost.** Two more routes (`/login`, `/logout`) and a cookie. The cookie cannot be revoked
without changing the token, there is one shared secret and no accounts, and over plain http
the token still crosses the network in the clear. The compose health check moved from
`/folders` to `/config`, which stays open. The dev tasks that call the API without a
header (`task dev:e2e`, `task smoke`) need the token when it is set.

**Benefit.** A shared instance can be published without a proxy in front, and the same
browser flow covers fetch, XHR and download links.

## Alternatives considered

**Bearer header only, token in `sessionStorage`.** One endpoint fewer, but the ZIP buttons
would become fetch-and-blob, losing the browser's download handling, and any script on the
page could read the token. The cookie is `HttpOnly`.

**Storing the token itself in the cookie.** Simpler, and a stolen cookie would then be the
token. The HMAC value is useless against the MCP port or another deployment.

**Accounts or OAuth.** Far more machinery than the rest of the service for a tool one
person or a small team runs.
