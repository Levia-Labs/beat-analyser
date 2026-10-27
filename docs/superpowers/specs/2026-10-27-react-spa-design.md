# React SPA frontend

- **Date:** 2026-10-27
- **Status:** Approved design

## Goal

Replace the Bootstrap page with a compact, mobile-friendly React single-page app in the
saber-tooth visual style, with the beat-analyser logo, and add a per-project view of the
analysed events so results are visible without unzipping.

## Decisions (from the user)

1. Scope: today's functions (file upload, YouTube URL, project list with ZIP download) plus a
   track viewer per project. No audio playback.
2. Serving: Flask serves the built SPA from the same origin. Source in `web/`, built by a Node
   stage in `app/Dockerfile`. One container; Node never runs at runtime.
3. Visuals: saber-tooth's palette, radius and font stack; the beat-analyser logo in the header.
4. The web service is published on port 8002 by default (was 80), next to MCP on 8001.

## Facts the design rests on

1. The current page is `app/templates/index.html`, `app/static/app.js` and `app/static/app.css`:
   Bootstrap 5.3.3 from jsDelivr with SRI, two forms, a project accordion, a loading modal.
2. `POST /upload` and `POST /download_yt` block until the project is built and return the ZIP
   (`app/app.py:104`, `app/app.py:131`). The MCP jobs (`app/backend.py`) and `tasks/dev/e2e.py`
   rely on that response.
3. The CSP today allows `https://cdn.jsdelivr.net` for scripts and styles (`app/app.py:27-33`).
4. saber-tooth's React frontend lives on its `react` worktree,
   `.worktrees/react/frontend` in the saber-tooth repository: React 19, Vite, TypeScript, plain CSS
   (`src/styles/tokens.css`, `base.css`), its own SVG icons, a `Logo.tsx`, vitest. Tokens: stage
   `#110e1f`, panel `#1a1630`, panel-2 `#231e40`, line `#2c2650`, red `#e0303a`, blue `#2e9be6`,
   ink `#ece9f7`, muted `#a39fbf`, amber `#e8a33d`, green `#3dd68c`, radius 8px, dark only, font
   stack `Bahnschrift, "DIN Alternate", "DIN 2014", system-ui, sans-serif`, breakpoints 719px and
   1099px.
5. The logo is `docs/logo.svg`.
6. The host has Node 22 (v22.14.0). The shared Taskfiles already provide `lint:web:check` and
   `lint:web:run` (eslint with zero warnings, prettier).
7. React sets inline styles through the CSSOM, which a CSP without `'unsafe-inline'` in
   `style-src` permits; only `style=""` attributes in parsed HTML are blocked.

## Architecture

| Path                                     | Role                                                                                               |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `web/`                                   | The SPA source: `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/`            |
| `web/src/styles/`                        | `tokens.css` (copied from saber-tooth, source named in a comment), `base.css`, per-screen CSS      |
| `web/src/api.ts`                         | Typed `fetch` wrappers for the endpoints below; one place that knows URLs                          |
| `web/src/screens/Home.tsx`, `Viewer.tsx` | The two routes                                                                                     |
| `web/src/components/`                    | `Logo`, `AnalyseCard`, `ProjectList`, `Timeline`, `Banner`                                         |
| `app/Dockerfile`                         | New first stage `node:22-slim`: `npm ci`, `npm run build`; runtime copies `web/dist` to `/app/web` |
| `app/app.py`                             | Serves `index.html` for `/` and `/p/<name>`, assets under `/assets/`; CSP tightened                |
| `app/events.py`                          | Shared CSV and labels reading, moved out of `app/mcp_tools.py`                                     |

1. Dependencies: `react`, `react-dom`, `react-router`. Data through small hooks over `fetch`;
   no query library, no UI kit, no CSS framework.
2. `WEB_DIST` (default `web/dist` locally, `/app/web` in the image) tells Flask where the build
   is. When it has no `index.html`, `/` answers 503 with a one-line hint to run `task web:build`,
   so the API still works in a checkout without Node.
3. Routes: `/` home, `/p/<name>` viewer. Flask returns `index.html` for both; it does not
   validate `<name>` there, since the SPA asks the API, which does.
4. Templates and `app/static/` are deleted. The `template-no-inline` ast-grep rule stays (it
   matches nothing until a template returns).
5. CSP: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;
connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none';
form-action 'self'`. jsDelivr is gone.
6. Port: `docker-compose.yml` publishes `${BIND:-127.0.0.1}:${PORT:-8002}:8000`. `task docker:up`
   prints `http://localhost:8002`. `task run` keeps 5000. `task web:dev` runs Vite on 5173 and
   proxies `/upload`, `/download_yt`, `/folders`, `/download`, `/projects` to
   `http://127.0.0.1:5000`.

## Visuals

1. Dark only, saber-tooth tokens. Panels on the stage colour, 1px `--line` borders, 8px radius,
   red for the primary action, blue for links and focus rings, amber for "analysing", green for
   done, red text for errors.
2. Header: logo 40px, radius 10px, wordmark "Beat Analyser", left-aligned, 56px tall. Favicon is
   the same SVG.
3. Compact: content column max 720px; 12px gaps; body text 15px; tabular numerals for times and
   BPM.
4. Mobile (max-width 719px): edge-to-edge cards, full-width controls, 44px minimum touch targets,
   the project list as stacked rows, the viewer's lane labels collapse to 2-letter tags.
5. `prefers-reduced-motion`: no transitions or progress animation.

## Screens

### Home `/`

1. **New analysis card** with two tabs, File and YouTube.
   1. File: a drop zone that is also a button opening the picker; accepted types from
      `GET /config` (below). Shows the chosen file name and size; one "Analyse" button.
   2. YouTube: one URL input, one "Analyse" button.
   3. While running: upload progress bar (XMLHttpRequest `upload.onprogress`), then an
      indeterminate "Analysing" state with elapsed time. The inputs are disabled.
   4. On success: navigate to `/p/<name>`.
   5. On error: a banner inside the card with the server's `error` text; the form stays filled
      for a retry.
2. **Project list**: one row per project, name, key, BPM, a "ZIP" button (link to
   `/download/<name>`), the row opens the viewer. Projects without a ZIP show "not built". Empty
   state: one muted line.

### Viewer `/p/<name>`

1. Header row: back link, name, key, BPM, "ZIP" button.
2. Timeline: one lane per track in the order kick, snare, hh, beats, bass, chords, melody, each a
   canvas row 28px tall (24px on mobile) with a label column; a time ruler above. Percussion and
   beats draw ticks; pitched tracks draw ticks whose vertical position follows the MIDI pitch
   within the lane.
3. Zoom: "-" and "+" buttons, Ctrl+wheel, pinch on touch. Horizontal scroll in a single
   scroll container shared by all lanes. Zoom range: whole song to 2 seconds per viewport.
4. Tapping or clicking a lane label shows its event count and first and last time below the
   timeline.
5. Not found: "No project called <name>" and a link home.

## API changes

1. `POST /upload` and `POST /download_yt`: when the request's `Accept` header prefers
   `application/json` (`request.accept_mimetypes.best == "application/json"`), the success answer
   is `200 {"name": "<project>"}` instead of the ZIP. Errors are JSON already. Default behaviour
   is unchanged, so curl, the MCP jobs and `dev:e2e` keep receiving the ZIP.
2. `GET /projects/<name>/events`: `200 {"name", "meta", "labels": {id: label}, "tracks": {track:
[[time, label_id], ...]}}`, built with `events.py`. `404 {"error"}` when the name fails
   `storage.is_valid_name` or the project has no ZIP yet (same "done" rule as MCP).
3. `GET /config`: `200 {"extensions": [...], "max_upload_mb": n}` so the SPA does not hard-code
   what the server accepts.
4. `/folders` and `/download/<name>` are unchanged.
5. `events.py` holds `tracks(folder)`, `rows(path)`, `labels(folder)`, `summary(folder)` and
   `meta(folder)`, moved from `mcp_tools.py` (which imports them). Flask never imports `mcp`.

## Error handling

1. Network failure or a non-JSON error page: the banner says "Server unreachable" or quotes the
   status.
2. Upload larger than `max_upload_mb`: refused in the browser before sending, with the same
   wording as the server's 413.
3. The events endpoint for a project still building: 404; the viewer shows "Not analysed yet"
   with a reload button.

## Testing

1. Frontend (vitest, Testing Library, jsdom): AnalyseCard states (idle, uploading, analysing,
   error, success navigates), size refusal, ProjectList rendering and empty state, Timeline
   scale maths (time to x, zoom bounds) as pure functions, api.ts error mapping.
2. Python (pytest): JSON answers for both POSTs and the unchanged ZIP default; `/config`;
   `/projects/<name>/events` happy path, traversal names, not-built 404; `/` and `/p/<name>`
   serve the SPA, 503 without a build; the new CSP header; `events.py` moved without behaviour
   change (the MCP tests keep passing).
3. `tasks/dev/e2e.py`: also GETs `/` and expects the SPA's `index.html` (a `<div id="root">`),
   then GETs the first `/assets/*.js` it references and expects 200.
4. `task docker:e2e` is required: the Dockerfile gains a stage.
5. `task validate` gains `web:test` and the shared `lint:web:check` over `web/`.

## Tasks

1. `web:install` (`npm ci` in `web/`), `web:dev`, `web:build`, `web:test`. Rows in
   `docs/tasks.md` and the Taskfile README.
2. `validate` runs `web:test` and `lint:web:check` when `web/node_modules` exists, and says it
   skipped them otherwise, as the Rust gates do without a toolchain.

## Docs

1. `docs/architecture.md`: the SPA and how Flask serves it.
2. `docs/api.md`: the JSON variant of both POSTs, `/projects/<name>/events`, `/config`.
3. `docs/configuration.md`: `PORT` default 8002 in compose, `WEB_DIST`.
4. `docs/development.md`: `task web:install`, `task web:dev`.
5. `docs/security.md`: the CSP row without jsDelivr; the SRI row removed.
6. `README.md` and `tasks/docker/hub.md`: the new port and a screenshot-free description.
7. An ADR: React SPA served by Flask from a Node build stage, over a separate web container.

## Out of scope

1. Audio playback.
2. A light theme.
3. Deleting or renaming projects from the UI.
4. Live progress from the server during an analysis.
