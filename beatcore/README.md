# beatcore

Onsets, percussion classes, tempo and beats from a WAV file, in Rust. Built as a Python
extension (PyO3, one abi3 wheel for CPython 3.12+) and as a debugging CLI. Why it exists:
[ADR 0009](../docs/adr/0009-beatcore-and-basic-pitch.md). The algorithms:
[docs/architecture.md](../docs/architecture.md#analysis).

```python
import beatcore

beatcore.analyse("audio.wav")
# {"bpm": 120.02, "confidence": 0.41, "beats": [0.5, 1.0, ...],
#  "onsets": [(0.01, "kick"), (0.25, "hh"), ...]}
```

| File                 | Role                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------ |
| `src/audio.rs`       | WAV in (hound), mixed to mono                                                        |
| `src/features.rs`    | Streaming STFT at 100 fps: SuperFlux envelope, band flux, beat envelope              |
| `src/onset.rs`       | Peak picking; kick / snare / hi-hat classification                                   |
| `src/tempo.rs`       | Tempo by autocorrelation with a 120 BPM prior; beats by dynamic programming          |
| `src/lib.rs`         | `analyse_file`, `analyse_samples`, and the Python module behind the `python` feature |
| `src/main.rs`        | `beatcore-cli FILE.wav`                                                              |
| `tests/synthetic.rs` | Drum loops and bursts with known tempo, beats and classes                            |

No host toolchain needed: `task rust:test`, `task rust:check` and
`task rust:analyse FILE=...` run in the pinned `rust` image. The Docker image builds the
wheel itself; `task deps:beatcore` installs it into `.venv` when a host `cargo` exists.
