# .basedpyright

Holds `baseline.json`, the basedpyright baseline: type errors already present, which `task types:check` tolerates. It
ratchets down only, so a new error fails the gate and a fixed one is removed from the file
([ADR 0006](../docs/adr/0006-ratcheting-baselines.md)). The namespace is described in
[tasks/types](../tasks/types/README.md).
