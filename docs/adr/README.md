# Architecture decision records

Decisions whose reasoning is not visible in the code. Someone will eventually look at one
of these and see an obvious improvement; the record is what stops the fix from being undone.

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-docker-is-the-runtime.md) | Docker is the supported runtime; the host venv is for development | Accepted |
| [0002](0002-madmom-from-git.md) | madmom is installed from a pinned git commit | Superseded by 0009 |
| [0003](0003-madmom-and-ffmpeg-only.md) | madmom and ffmpeg do all audio I/O; librosa and pydub removed | Superseded by 0009 |
| [0004](0004-yt-dlp-unpinned.md) | yt-dlp is the one unpinned dependency | Accepted |
| [0005](0005-name-keyed-projects.md) | Projects are keyed by name and the ZIP is the cache | Accepted |
| [0006](0006-ratcheting-baselines.md) | Quality gates ratchet against a committed baseline | Accepted |
| [0007](0007-taskfile-as-the-interface.md) | go-task is the interface for people and agents | Accepted |
| [0008](0008-loopback-without-auth.md) | No authentication; loopback by default, same-origin POSTs | Accepted |
| [0009](0009-beatcore-and-basic-pitch.md) | beatcore (Rust) and basic-pitch (ONNX) replace madmom | Accepted |
| [0010](0010-two-tier-local-pipeline.md) | Verification runs locally in two tiers, as tasks | Accepted |

## Writing one

One file per decision, `NNNN-short-title.md`, numbered in order. Copy
[template.md](template.md). Write one when someone would reasonably do the opposite and the
reason not to is not in the code, when a removal is the decision, or when a cost was
accepted knowingly. A decision that turns out wrong is superseded, not deleted.
