# tasks / eval

Namespace `eval:`. Scores beatcore against ground truth, inside the app image (where
beatcore is built): `task docker:build` first after a change under `beatcore/`.

| Script | Task | Role |
| --- | --- | --- |
| `eval.py` | `task eval:synthetic [SECONDS=20]` | Five generated styles (four-on-the-floor, backbeat, swing, syncopated, pad) at 70-180 BPM with known beats and drum hits |
| `eval.py` | `task eval:synthetic:diff [THRESHOLD=0.05]` | The tuning loop: means with their change and only the clip scores that moved, against the baseline in `.cache/` (gitignored); `task eval:synthetic:baseline` accepts the last run |
| `eval.py` | `task eval:dir DIR=path` | Any folder of audio with `<stem>.beats` (Ballroom annotation format) and optional `<stem>.bpm` |

Metrics, as in MIREX: `acc1` tempo within 4 percent; `acc2` also accepting x2, x1/2, x3,
x1/3; `beatF` beat F-measure at 70 ms; per-class onset F-measure at 50 ms (synthetic only).

Tune `PRIOR_BPM` (`beatcore/src/tempo.rs`), the peak-picking `delta`
(`beatcore/src/onset.rs`) and `BEAT_WEIGHTS` (`beatcore/src/features.rs`) against these
numbers, not by ear. A synthetic score is a floor: real music is harder.
