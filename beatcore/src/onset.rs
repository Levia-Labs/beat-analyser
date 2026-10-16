//! Peak picking on the onset envelope, and kick / snare / hi-hat classification.

/// Percussion classes, in the order of `Features::band_flux`.
pub const LABELS: [&str; 3] = ["kick", "snare", "hh"];

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

/// Per-bin density weights for [kick, snare, hh]. The hi-hat weight decides ties: white
/// noise rises equally per bin in the snare and hi-hat bands, so a flat attack is a hi-hat
/// unless the mid band clearly dominates - as a snare's body around 200 Hz makes it.
const DENSITY_WEIGHTS: [f32; 3] = [0.3, 1.0, 1.5];

/// Per-bin rise per band over the onset frame and the two after it (the attack).
fn attack(band_flux: &[[f32; 3]], frame: usize) -> [f32; 3] {
    let end = (frame + 3).min(band_flux.len());
    let mut sums = [0.0f32; 3];
    for bands in &band_flux[frame..end] {
        for (sum, value) in sums.iter_mut().zip(bands) {
            *sum += value;
        }
    }
    sums
}

/// One label per onset: the band with the largest weighted rise per bin.
///
/// What this replaced, measured with `task eval:synthetic` (kick / snare / hh F-measure):
///   1. Raw band sums: the hi-hat band holds ~80 percent of the bins and won every noisy
///      hit. Snare 0.17.
///   2. Sums over each band's mean across the song, with co-labels: 0.85 / 0.67 / 0.91,
///      but a song with no snare has a tiny snare mean, so every white-noise hat also read
///      as a snare (ten false snares in `task docker:e2e`'s fixture).
///   3. Sums over each band's leakage level (25th percentile over onsets): fixed that,
///      but leakage levels are not comparable between bands. 0.65 / 0.88 / 0.74, and with
///      a co-label rule on top 0.68 / 0.34 / 0.77.
pub fn classify_all(band_flux: &[[f32; 3]], peaks: &[usize]) -> Vec<(usize, &'static str)> {
    peaks
        .iter()
        .filter_map(|&frame| {
            let rise = attack(band_flux, frame);
            let best = (0..3).max_by(|&a, &b| {
                (rise[a] * DENSITY_WEIGHTS[a])
                    .partial_cmp(&(rise[b] * DENSITY_WEIGHTS[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            (rise[best] > 0.0).then_some((frame, LABELS[best]))
        })
        .collect()
}
