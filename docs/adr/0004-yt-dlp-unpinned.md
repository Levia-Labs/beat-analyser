# 0004. yt-dlp is the one unpinned dependency

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

Every other dependency is pinned. yt-dlp talks to a site that changes without notice; a
release a few weeks old regularly stops working for some videos.

## Decision

`yt-dlp` appears unversioned in `app/requirements.txt`. `task deps:yt-dlp` upgrades the host
venv; `task docker:rebuild` gets the newest release into the image.

## Consequences

**Cost.** Builds are not byte-reproducible, and a bad yt-dlp release lands on rebuild
without review. The supply-chain exposure is a package with a long, active history.

**Benefit.** YouTube downloads keep working with a rebuild rather than a pin bump.

## Alternatives considered

**Pin and bump on breakage.** Every breakage becomes an edit, a commit and a rebuild instead
of a rebuild.

**Drop YouTube support.** Upload-only would remove the dependency entirely. Kept because the
feature is used.
