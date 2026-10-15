//! Peak picking on the onset envelope, and kick / snare / hi-hat classification.

/// Percussion classes, in the order of `Features::band_flux`.
pub const LABELS: [&str; 3] = ["kick", "snare", "hh"];
/// Band weights carried over from the madmom version, tuned by ear there.
const WEIGHTS: [f32; 3] = [1.2, 1.1, 1.0];

pub struct PeakParams {
    /// Seconds a peak must dominate on each side.
    pub max_window: f64,
    /// Seconds before and after averaged for the adaptive threshold.
    pub pre_avg: f64,
    pub post_avg: f64,
    /// Margin above the local average, as a fraction of the envelope's maximum.
    pub delta: f32,
    /// Onsets closer than this, in seconds, are merged into the first.
    pub combine: f64,
}

impl Default for PeakParams {
    fn default() -> Self {
        Self {
            max_window: 0.03,
            pre_avg: 0.10,
            post_avg: 0.07,
            delta: 0.05,
            combine: 0.03,
        }
    }
}

/// Frame indices of onsets: local maxima above the local mean plus `delta`.
pub fn pick_peaks(env: &[f32], fps: f64, params: &PeakParams) -> Vec<usize> {
    let peak = env.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let frames = |seconds: f64| (seconds * fps).round() as usize;
    let max_w = frames(params.max_window).max(1);
    let (pre_avg, post_avg) = (frames(params.pre_avg), frames(params.post_avg));
    let combine = frames(params.combine);
    let n = env.len();

    let mut found = Vec::new();
    for t in 0..n {
        let value = env[t] / peak;
        if value <= 0.0 {
            continue;
        }
        let lo = t.saturating_sub(max_w);
        let hi = (t + max_w).min(n - 1);
        if env[lo..=hi].iter().any(|&v| v > env[t]) {
            continue;
        }
        let a_lo = t.saturating_sub(pre_avg);
        let a_hi = (t + post_avg).min(n - 1);
        let mean = env[a_lo..=a_hi].iter().sum::<f32>() / (a_hi - a_lo + 1) as f32 / peak;
        if value < mean + params.delta {
            continue;
        }
        if let Some(&last) = found.last() {
            if t - last <= combine {
                continue;
            }
        }
        found.push(t);
    }
    found
}

/// Which band rose most at the onset, over the onset frame and the two after it (the attack).
pub fn classify(band_flux: &[[f32; 3]], frame: usize) -> Option<&'static str> {
    let end = (frame + 3).min(band_flux.len());
    let mut sums = [0.0f32; 3];
    for bands in &band_flux[frame..end] {
        for (sum, value) in sums.iter_mut().zip(bands) {
            *sum += value;
        }
    }
    let total: f32 = sums.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let best = (0..3)
        .max_by(|&a, &b| {
            (sums[a] * WEIGHTS[a])
                .partial_cmp(&(sums[b] * WEIGHTS[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    Some(LABELS[best])
}
