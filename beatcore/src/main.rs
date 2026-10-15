//! beatcore-cli FILE.wav - print tempo, beats and labelled onsets. For debugging the crate
//! without Python.

use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: beatcore-cli FILE.wav");
        return ExitCode::from(2);
    };
    match beatcore::analyse_file(std::path::Path::new(&path)) {
        Ok(a) => {
            println!("bpm {:.2}  confidence {:.3}", a.bpm, a.confidence);
            println!("beats {}", a.beats.len());
            for label in beatcore::onset::LABELS {
                let n = a.onsets.iter().filter(|(_, l)| *l == label).count();
                println!("{label:<6} {n}");
            }
            for (t, label) in &a.onsets {
                println!("{t:.3},{label}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
