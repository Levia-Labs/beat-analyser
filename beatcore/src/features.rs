//! One streaming pass of short-time Fourier transforms at 100 frames per second.
//!
//! Nothing is kept per frame except what later stages read: one full-band onset value and
//! three band onset values. A five-minute song is 30 000 frames; keeping whole spectra would
//! cost over 100 MB.

use realfft::RealFftPlanner;

/// Frames per second of every envelope this crate produces.
pub const FPS: f64 = 100.0;

/// Band edges in Hz: kick up to 150, snare 150 to 4000, hi-hat above.
pub const KICK_MAX_HZ: f64 = 150.0;
pub const SNARE_MAX_HZ: f64 = 4000.0;

/// Log compression `ln(1 + LAMBDA * |X|)`, as in madmom's spectral-flux family.
const LAMBDA: f32 = 100.0;

pub struct Features {
    /// Frames per second actually achieved (sample rate / integer hop).
    pub fps: f64,
    /// Full-band log spectral flux with a frequency max-filter on the reference frame
    /// (SuperFlux): vibrato and slow pitch changes do not register as onsets.
    pub flux: Vec<f32>,
    /// Positive change in linear magnitude, summed per band: [kick, snare, hh]. What is new
    /// in each band at a frame, used to classify the onset there.
    pub band_flux: Vec<[f32; 3]>,
    /// The full-band flux split into the same three bands, for `beat_envelope`.
    pub band_log_flux: Vec<[f32; 3]>,
}

/// Weights of the kick, snare and hi-hat bands in the beat envelope.
const BEAT_WEIGHTS: [f32; 3] = [1.0, 0.7, 0.35];

impl Features {
    /// The envelope tempo and beats are tracked on. Each band is normalised by its own mean
    /// and weighted toward the low end: in the full-band flux every bin counts equally, so
    /// broadband hi-hats on the off-beat outweighed a kick that fills a few low bins, and the
    /// tracker put every beat half a beat late.
    pub fn beat_envelope(&self) -> Vec<f32> {
        let n = self.band_log_flux.len().max(1) as f32;
        let mut means = [0.0f32; 3];
        for bands in &self.band_log_flux {
            for (m, v) in means.iter_mut().zip(bands) {
                *m += v / n;
            }
        }
        self.band_log_flux
            .iter()
            .map(|bands| {
                (0..3)
                    .filter(|&b| means[b] > 0.0)
                    .map(|b| BEAT_WEIGHTS[b] * bands[b] / means[b])
                    .sum()
            })
            .collect()
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n as f32;
            x.sin().powi(2)
        })
        .collect()
}

pub fn extract(samples: &[f32], sample_rate: u32) -> Features {
    let hop = ((sample_rate as f64 / FPS).round() as usize).max(1);
    let n = if sample_rate >= 32_000 { 2048 } else { 1024 };
    let window = hann(n);
    let bin_hz = sample_rate as f64 / n as f64;
    let bins = n / 2 + 1;
    let kick_end = ((KICK_MAX_HZ / bin_hz).floor() as usize + 1).min(bins);
    let snare_end = ((SNARE_MAX_HZ / bin_hz).floor() as usize + 1).min(bins);

    // Frames are centred: frame t covers samples around t * hop.
    let pad = n / 2;
    let n_frames = samples.len() / hop + 1;
    let sample_at = |i: isize| -> f32 {
        if i < 0 {
            0.0
        } else {
            samples.get(i as usize).copied().unwrap_or(0.0)
        }
    };

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let mut prev_mag = vec![0.0f32; bins];
    let mut prev_log_max = vec![0.0f32; bins];
    let mut mag = vec![0.0f32; bins];
    let mut log = vec![0.0f32; bins];
    let mut flux = Vec::with_capacity(n_frames);
    let mut band_flux = Vec::with_capacity(n_frames);
    let mut band_log_flux = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let start = (frame * hop) as isize - pad as isize;
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = sample_at(start + i as isize) * window[i];
        }
        fft.process(&mut input, &mut output)
            .expect("buffer sizes come from the same plan");

        for k in 0..bins {
            mag[k] = output[k].norm();
            log[k] = (1.0 + LAMBDA * mag[k]).ln();
        }

        let mut full = 0.0f32;
        let mut bands = [0.0f32; 3];
        let mut log_bands = [0.0f32; 3];
        for k in 0..bins {
            let log_rise = (log[k] - prev_log_max[k]).max(0.0);
            full += log_rise;
            let band = if k < kick_end {
                0
            } else if k < snare_end {
                1
            } else {
                2
            };
            bands[band] += (mag[k] - prev_mag[k]).max(0.0);
            log_bands[band] += log_rise;
        }
        // Before the first frame is silence (prev_* start at zero), so a hit at sample 0 is an
        // onset. Zeroing frame 0 instead lost the opening kick of every clip that starts on one.
        flux.push(full);
        band_flux.push(bands);
        band_log_flux.push(log_bands);

        // Reference for the next frame: max over three neighbouring bins.
        for (k, slot) in prev_log_max.iter_mut().enumerate() {
            let lo = k.saturating_sub(1);
            let hi = (k + 1).min(bins - 1);
            *slot = log[lo..=hi].iter().copied().fold(f32::MIN, f32::max);
        }
        prev_mag.copy_from_slice(&mag);
    }

    Features {
        fps: sample_rate as f64 / hop as f64,
        flux,
        band_flux,
        band_log_flux,
    }
}
