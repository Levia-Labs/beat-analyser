//! Global tempo from the autocorrelation of the onset envelope, and beat positions by
//! dynamic programming (Ellis 2007, "Beat Tracking by Dynamic Programming" - the method
//! librosa's `beat_track` implements).

/// Tempo search range in BPM.
pub const MIN_BPM: f64 = 40.0;
pub const MAX_BPM: f64 = 240.0;
/// Log-normal prior: centred on 120 BPM, one octave standard deviation. It is what resolves
/// the half / double tempo ambiguity every periodic signal has.
const PRIOR_BPM: f64 = 120.0;
const PRIOR_OCTAVES: f64 = 1.0;
/// How strongly beat spacing is held to the tempo. librosa's default.
const TIGHTNESS: f64 = 100.0;

pub struct Tempo {
    pub bpm: f64,
    /// Normalised autocorrelation at the chosen lag, 0 to 1.
    pub confidence: f64,
}

pub fn estimate(env: &[f32], fps: f64) -> Option<Tempo> {
    let mean = env.iter().sum::<f32>() / env.len().max(1) as f32;
    let x: Vec<f64> = env.iter().map(|&v| f64::from(v - mean)).collect();
    let min_lag = (60.0 * fps / MAX_BPM).floor() as usize;
    let max_lag = (60.0 * fps / MIN_BPM).ceil() as usize;
    if x.len() < max_lag + 2 || min_lag < 2 {
        return None;
    }

    let acf = |lag: usize| -> f64 { x.iter().zip(&x[lag..]).map(|(a, b)| a * b).sum() };
    let energy = acf(0);
    if energy <= 0.0 {
        return None;
    }
    let values: Vec<f64> = (min_lag - 1..=max_lag + 1).map(acf).collect();
    let at = |lag: usize| values[lag + 1 - min_lag];

    let prior = |lag: usize| {
        let bpm = 60.0 * fps / lag as f64;
        (-0.5 * ((bpm / PRIOR_BPM).log2() / PRIOR_OCTAVES).powi(2)).exp()
    };
    let best = (min_lag..=max_lag).max_by(|&a, &b| {
        (at(a) * prior(a))
            .partial_cmp(&(at(b) * prior(b)))
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    // Parabolic interpolation around the peak: sub-frame lag, so a tempo is not limited to
    // the 100 fps grid (50 frames at 120 BPM would otherwise mean 2.4 BPM steps).
    let (l, c, r) = (at(best - 1), at(best), at(best + 1));
    let curve = l - 2.0 * c + r;
    let shift = if curve < 0.0 {
        (0.5 * (l - r) / curve).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some(Tempo {
        bpm: 60.0 * fps / (best as f64 + shift),
        confidence: (c / energy).clamp(0.0, 1.0),
    })
}

/// Beat times in seconds for a known tempo.
pub fn track(env: &[f32], fps: f64, bpm: f64) -> Vec<f64> {
    let n = env.len();
    let period = 60.0 * fps / bpm;
    if n == 0 || period < 1.0 {
        return Vec::new();
    }

    // Local score: the envelope normalised by its standard deviation, smoothed with a
    // Gaussian a sixteenth of a beat wide.
    let mean = env.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let var = env
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt().max(1e-12);
    let reach = period.round() as isize;
    let kernel: Vec<f64> = (-reach..=reach)
        .map(|i| (-0.5 * (i as f64 * 32.0 / period).powi(2)).exp())
        .collect();
    let local: Vec<f64> = (0..n as isize)
        .map(|t| {
            kernel
                .iter()
                .enumerate()
                .filter_map(|(j, w)| {
                    let i = t + j as isize - reach;
                    (0..n as isize)
                        .contains(&i)
                        .then(|| w * f64::from(env[i as usize]) / std)
                })
                .sum()
        })
        .collect();

    let max_local = local.iter().copied().fold(0.0, f64::max);
    let near = (period / 2.0).round() as usize;
    let far = (2.0 * period).round() as usize;
    let mut cum = vec![0.0f64; n];
    let mut back = vec![usize::MAX; n];
    let mut first = true;

    for i in 0..n {
        let mut best: Option<(f64, usize)> = None;
        if i >= near {
            let from = i.saturating_sub(far);
            for (z, &previous) in cum[..=i - near].iter().enumerate().skip(from) {
                let gap = (i - z) as f64 / period;
                let score = previous - TIGHTNESS * gap.ln().powi(2);
                if best.is_none_or(|(s, _)| score > s) {
                    best = Some((score, z));
                }
            }
        }
        // Until the first real onset, every frame starts its own chain.
        if first && local[i] < 0.01 * max_local {
            cum[i] = local[i];
        } else {
            first = false;
            match best {
                Some((score, z)) => {
                    cum[i] = local[i] + score;
                    back[i] = z;
                }
                None => cum[i] = local[i],
            }
        }
    }

    // The last beat: the latest local maximum of the cumulative score that reaches half
    // the median of all of them.
    let maxima: Vec<usize> = (1..n.saturating_sub(1))
        .filter(|&i| cum[i] > cum[i - 1] && cum[i] >= cum[i + 1])
        .collect();
    if maxima.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = maxima.iter().map(|&i| cum[i]).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let Some(&last) = maxima.iter().rev().find(|&&i| cum[i] >= 0.5 * median) else {
        return Vec::new();
    };

    let mut beats = vec![last];
    while let Some(&prev) = beats.last().map(|&b| &back[b]) {
        if prev == usize::MAX {
            break;
        }
        beats.push(prev);
    }
    beats.reverse();

    // Trim weak beats at both ends: before the music starts and after it stops.
    let rms = (beats.iter().map(|&b| local[b].powi(2)).sum::<f64>() / beats.len() as f64).sqrt();
    let threshold = 0.5 * rms;
    let start = beats
        .iter()
        .position(|&b| local[b] >= threshold)
        .unwrap_or(0);
    let end = beats
        .iter()
        .rposition(|&b| local[b] >= threshold)
        .map_or(beats.len(), |p| p + 1);
    beats[start..end].iter().map(|&b| b as f64 / fps).collect()
}
