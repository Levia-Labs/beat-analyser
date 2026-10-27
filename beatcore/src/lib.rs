//! beatcore: onsets, percussion classes, tempo and beats from a WAV file.
//!
//! Replaces madmom's CNN onset detector in beat-analyser (docs/adr/0009). One STFT pass
//! feeds everything: a SuperFlux onset envelope for onsets, a low-weighted band envelope
//! for tempo and beats, and per-band flux for kick / snare / hi-hat classification.

pub mod audio;
pub mod features;
pub mod onset;
pub mod tempo;

use std::path::Path;

pub struct Analysis {
    pub bpm: f64,
    pub confidence: f64,
    pub beats: Vec<f64>,
    /// (seconds, "kick" | "snare" | "hh"), in time order. Simultaneous hits in different
    /// bands share a time and appear once per band.
    pub onsets: Vec<(f64, &'static str)>,
}

pub fn analyse_samples(samples: &[f32], sample_rate: u32) -> Analysis {
    let feats = features::extract(samples, sample_rate);
    let peaks = onset::pick_peaks(&feats.flux, feats.fps, &onset::PeakParams::default());
    let onsets = onset::classify_all(&feats.band_flux, &peaks)
        .into_iter()
        .map(|(f, l)| (f as f64 / feats.fps, l))
        .collect();

    let beat_env = feats.beat_envelope();
    let (bpm, confidence, beats) = match tempo::estimate(&beat_env, feats.fps) {
        Some(t) => {
            let beats = tempo::track(&beat_env, feats.fps, t.bpm);
            (t.bpm, t.confidence, beats)
        }
        None => (0.0, 0.0, Vec::new()),
    };
    Analysis {
        bpm,
        confidence,
        beats,
        onsets,
    }
}

pub fn analyse_file(path: &Path) -> Result<Analysis, String> {
    let audio = audio::read_wav(path)?;
    Ok(analyse_samples(&audio.samples, audio.sample_rate))
}

#[cfg(feature = "python")]
mod python {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    fn to_dict<'py>(py: Python<'py>, a: &crate::Analysis) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bpm", a.bpm)?;
        dict.set_item("confidence", a.confidence)?;
        dict.set_item("beats", a.beats.clone())?;
        dict.set_item("onsets", a.onsets.clone())?;
        Ok(dict)
    }

    /// Analyse a WAV file. Returns {"bpm", "confidence", "beats", "onsets"}.
    #[pyfunction]
    fn analyse<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
        // Detached so other Python threads (request handlers, progress polls) run meanwhile.
        let path = std::path::PathBuf::from(path);
        let analysis = py
            .detach(move || crate::analyse_file(&path))
            .map_err(PyValueError::new_err)?;
        to_dict(py, &analysis)
    }

    /// Analyse mono samples in [-1, 1]. Same result shape as `analyse`.
    #[pyfunction]
    fn analyse_samples<'py>(
        py: Python<'py>,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let analysis = py.detach(move || crate::analyse_samples(&samples, sample_rate));
        to_dict(py, &analysis)
    }

    #[pymodule]
    fn beatcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(analyse, m)?)?;
        m.add_function(wrap_pyfunction!(analyse_samples, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
