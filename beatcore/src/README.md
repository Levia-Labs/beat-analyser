# beatcore / src

The Rust crate behind the rhythm tracks. One STFT pass feeds onsets, percussion classes, tempo
and beats. Context and metrics are in [beatcore/README.md](../README.md).

| File          | What it does                                                                      |
| ------------- | --------------------------------------------------------------------------------- |
| `lib.rs`      | Crate root: `analyse_samples`, `analyse_file` and the Python module `beatcore`    |
| `main.rs`     | `beatcore-cli FILE.wav`: prints tempo, beats and labelled onsets, without Python  |
| `audio.rs`    | Reads PCM WAV only; ffmpeg has already decoded uploads                            |
| `features.rs` | Streaming STFT at 100 frames per second; full-band and three band onset envelopes |
| `onset.rs`    | Peak picking on the onset envelope and kick, snare and hi-hat classification      |
| `tempo.rs`    | Global tempo from envelope autocorrelation, beats by dynamic programming (Ellis)  |
