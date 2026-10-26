# .ratchet

Holds `long-files.baseline`: files already over the file-length limit, with their current line count. A listed file
may not grow and a new file may not join; the baseline ratchets down only ([ADR 0006](../docs/adr/0006-ratcheting-baselines.md)).
The checker is crepidoma's `ratchet.py`, run through `task quality:gate`.
