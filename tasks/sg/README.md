# tasks / sg

Namespace `sg:`. [ast-grep](https://ast-grep.github.io/) rules in `rules/`, one test file per
rule in `tests/`, wired by [sgconfig.yml](../../sgconfig.yml). A new rule needs a test listing
at least one `valid` and one `invalid` snippet; `task sg:test` runs them.

`task sg:search P='pattern'` is structural search: `$X` matches one node, `$$$X` any number.
It finds call shapes a text grep cannot (`subprocess.run($$$A)` across line breaks).
