//! Synthetic signals with known ground truth: tempo, beat times and percussion classes.

use beatcore::analyse_samples;

const SR: u32 = 44_100;

/// A decaying 55 Hz kick on every beat, with optional noise hi-hats half a beat later.
fn drum_loop(bpm: f64, seconds: f64, hats: bool) -> Vec<f32> {
    let beat = 60.0 / bpm;
    let mut seed = 12_345u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    (0..(seconds * SR as f64) as usize)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let since = t % beat;
            let kick = (2.0 * std::f64::consts::PI * 55.0 * t).sin() * (-30.0 * since).exp();
            let off = (t - beat / 2.0).rem_euclid(beat);
            let hat = if hats && t >= beat / 2.0 {
                f64::from(noise()) * (-80.0 * off).exp() * 0.5
            } else {
                0.0
            };
            (0.4 * (kick + hat)) as f32
        })
        .collect()
}

fn burst(freq: Option<f64>, at: &[f64], seconds: f64) -> Vec<f32> {
    let mut seed = 7u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    (0..(seconds * SR as f64) as usize)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let env: f64 = at
                .iter()
                .filter(|&&s| t >= s)
                .map(|&s| (-40.0 * (t - s)).exp())
                .sum();
            let carrier = match freq {
                Some(f) => (2.0 * std::f64::consts::PI * f * t).sin(),
                None => f64::from(noise()),
            };
            (0.4 * env * carrier) as f32
        })
        .collect()
}

fn hits(found: &[f64], truth: &[f64], tolerance: f64) -> usize {
    truth
        .iter()
        .filter(|&&t| found.iter().any(|&f| (f - t).abs() <= tolerance))
        .count()
}

#[test]
fn tempo_of_drum_loops() {
    for bpm in [90.0, 120.0, 140.0] {
        let a = analyse_samples(&drum_loop(bpm, 12.0, true), SR);
        assert!(
            (a.bpm - bpm).abs() < 1.5,
            "expected {bpm} BPM, got {:.2}",
            a.bpm
        );
        assert!(a.confidence > 0.1, "confidence {:.3}", a.confidence);
    }
}

#[test]
fn beats_land_on_kicks() {
    let bpm = 120.0;
    let a = analyse_samples(&drum_loop(bpm, 12.0, true), SR);
    let truth: Vec<f64> = (1..23).map(|k| k as f64 * 0.5).collect();
    let found = hits(&a.beats, &truth, 0.07);
    assert!(
        found as f64 >= 0.9 * truth.len() as f64,
        "{found} of {} beats within 70 ms; beats {:?}",
        truth.len(),
        a.beats
    );
}

#[test]
fn onsets_found_and_classified() {
    let a = analyse_samples(&drum_loop(120.0, 6.0, true), SR);
    let kicks: Vec<f64> = a
        .onsets
        .iter()
        .filter(|o| o.1 == "kick")
        .map(|o| o.0)
        .collect();
    let hats: Vec<f64> = a
        .onsets
        .iter()
        .filter(|o| o.1 == "hh")
        .map(|o| o.0)
        .collect();
    let kick_truth: Vec<f64> = (0..12).map(|k| k as f64 * 0.5).collect();
    let hat_truth: Vec<f64> = (0..11).map(|k| 0.25 + k as f64 * 0.5).collect();
    // All twelve, the one at 0.0 s included.
    assert_eq!(hits(&kicks, &kick_truth, 0.03), 12, "kicks {kicks:?}");
    assert!(hits(&hats, &hat_truth, 0.03) >= 10, "hats {hats:?}");
}

#[test]
fn bands_classify_bursts() {
    let at = [0.5, 1.0, 1.5, 2.0];
    for (freq, label) in [(Some(60.0), "kick"), (Some(1000.0), "snare"), (None, "hh")] {
        let a = analyse_samples(&burst(freq, &at, 2.5), SR);
        let labelled: Vec<_> = a.onsets.iter().filter(|o| o.1 == label).collect();
        assert!(
            labelled.len() >= 3,
            "{label}: expected bursts at {at:?}, got {:?}",
            a.onsets
        );
    }
}

#[test]
fn silence_is_empty() {
    let a = analyse_samples(&vec![0.0; SR as usize * 3], SR);
    assert!(a.onsets.is_empty());
    assert!(a.beats.is_empty());
}

#[test]
fn short_input_does_not_panic() {
    let a = analyse_samples(&[0.1, -0.1, 0.2], SR);
    assert!(a.onsets.len() <= 1);
}
