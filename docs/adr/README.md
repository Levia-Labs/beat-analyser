# Architecture decision records

Decisions whose reasoning is not visible in the code. Someone will eventually look at one
of these and see an obvious improvement; the record is what stops the fix from being undone.

| #                                         | Decision                                                          | Status                           |
| ----------------------------------------- | ----------------------------------------------------------------- | -------------------------------- |
| [0001](0001-docker-is-the-runtime.md)     | Docker is the supported runtime; the host venv is for development | Accepted                         |
| [0002](0002-madmom-from-git.md)           | madmom is installed from a pinned git commit                      | Superseded by 0009               |
| [0003](0003-madmom-and-ffmpeg-only.md)    | madmom and ffmpeg do all audio I/O; librosa and pydub removed     | Superseded by 0009               |
| [0004](0004-yt-dlp-unpinned.md)           | yt-dlp is the one unpinned dependency                             | Accepted                         |
| [0005](0005-name-keyed-projects.md)       | Projects are keyed by name and the ZIP is the cache               | Superseded by 0014               |
| [0006](0006-ratcheting-baselines.md)      | Quality gates ratchet against a committed baseline                | Superseded by Crepidoma ADR-0005 |
| [0007](0007-taskfile-as-the-interface.md) | go-task is the interface for people and agents                    | Superseded by Crepidoma ADR-0004 |
| [0008](0008-loopback-without-auth.md)     | No authentication; loopback by default, same-origin POSTs         | Accepted                         |
| [0009](0009-beatcore-and-basic-pitch.md)  | beatcore (Rust) and basic-pitch (ONNX) replace madmom             | Accepted                         |
| [0010](0010-two-tier-local-pipeline.md)   | Verification runs locally in two tiers, as tasks                  | Superseded by Crepidoma ADR-0002 |
| [0011](0011-tool-first-agent-loop.md)     | Agents pick tools; tools return only what the next decision needs | Superseded by Crepidoma ADR-0001 |
| [0012](0012-mcp-as-a-sibling-proxy.md)    | MCP as a sibling service that proxies to the web app              | Accepted                         |
| [0013](0013-react-spa-served-by-flask.md) | React SPA built in a Node stage and served by Flask               | Accepted                         |
| [0014](0014-content-hash-projects.md)     | A name collision compares the audio hash, and analyses are capped | Accepted                         |
| [0015](0015-optional-web-token.md)        | An optional web token, held by the browser as a session cookie    | Accepted                         |

## Writing one

One file per decision, `NNNN-short-title.md`, numbered in order. Copy
[template.md](template.md). Write one when someone would reasonably do the opposite and the
reason not to is not in the code, when a removal is the decision, or when a cost was
accepted knowingly. A decision that turns out wrong is superseded, not deleted.
