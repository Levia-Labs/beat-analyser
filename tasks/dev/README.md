# tasks / dev

Namespace `dev:`, plus the scripts behind root `task smoke`, `task gate` and `task outline`. Each script turns something done
by hand once into one allowlisted `task` call, so later sessions run it without an approval
prompt. All are standard-library Python and run on the host's `python3`.

| Script | Task | Role |
| --- | --- | --- |
| `gate.py` | `task gate T="a,b K=x" [LINES=]` | Run tasks as gates: `ok`/`FAIL` line each, a failure's decisive lines (errors, panics, asserts, file:line) without build noise, full log in `.cache/gate/` |
| `outline.py` | `task outline F="path ..."` | Python defs and classes with end lines, Rust items, Markdown headings, Taskfile tasks, JS functions, HTML ids, each with its line number |
| `smoke.py` | `task smoke [URL=]` | Probe a running server: index headers, `/folders`. Uploads nothing |
| `fixture.py` | `task dev:fixture [NAME=] [SECONDS=] [BPM=]` | Synthetic clip with known kicks, hats and an A4 note, WAV and MP3, in `.cache/fixtures/` |
| `e2e.py` | `task dev:e2e [URL=]`, `task docker:e2e [PORT=]` | Upload a fixture, check the ZIP, the cross-origin refusal and the foreign-URL refusal |
| `encoding.py` | `task dev:encoding`, `task dev:encoding:fix` | Find or fix tracked text files that are not UTF-8 without a BOM |
| `packages.py` | `task dev:packages GITHUB="o/r" PYPI="pkg" CRATES="crate"` | Stars, licence and activity from GitHub; version, wheels and dependencies from PyPI; latest version and downloads from crates.io |
| `basic_pitch.py` | `task dev:basic-pitch [VERSION=0.4.0]` | Fetch basic-pitch's ONNX model and licence from its PyPI wheel, sha256-verified, into `app/models/`; upstream sources into `.cache/basic-pitch-src/` for reference |

Fixture projects are named `fixture-*`; `task uploads:fixtures:clean` removes them and
nothing else.
