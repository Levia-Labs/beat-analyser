# web

Tasks for the React SPA in `web/`. npm runs inside `web/`; nothing is installed at the
repository root. `web:check` is part of `task validate` and skips with a note when
`web/node_modules` is missing.

| Task          | What it does                                                             |
| ------------- | ------------------------------------------------------------------------ |
| `web:install` | `npm ci` from `package-lock.json`                                        |
| `web:lock`    | `npm install`, rewriting `package-lock.json` after a `package.json` edit |
| `web:dev`     | Vite dev server on :5173, API proxied to `task run` on :5000             |
| `web:build`   | Type-check and build `web/dist`, which Flask serves                      |
| `web:test`    | vitest over `web/src`                                                    |
| `web:format`  | Rewrite `web/` with prettier and the shared config                       |
| `web:check`   | Typecheck, eslint, prettier check and vitest                             |
