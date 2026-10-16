"""Musical key from transcribed notes: Krumhansl-Schmuckler key finding.

A pitch-class histogram, each note weighted by duration times amplitude, is correlated with
the Krumhansl-Kessler major and minor profiles in all twelve rotations; the best correlation
names the key. No model, no audio: it reads the notes basic-pitch already produced.
"""

import numpy as np

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Below this correlation the histogram fits no key well (drums only, one note): say so.
MIN_CORRELATION = 0.5


def histogram(notes) -> np.ndarray:
    """notes: (start s, end s, MIDI pitch, amplitude)."""
    weights = np.zeros(12)
    for start, end, pitch, amp in notes:
        weights[int(pitch) % 12] += max(end - start, 0.0) * amp
    return weights


def estimate(notes) -> tuple[str, float]:
    """("A minor", correlation), or ("unknown", correlation) when nothing fits."""
    weights = histogram(notes)
    if np.count_nonzero(weights) < 3:
        return "unknown", 0.0
    best_name, best_r = "unknown", -1.0
    for tonic in range(12):
        for mode, profile in (("major", MAJOR), ("minor", MINOR)):
            r = float(np.corrcoef(weights, np.roll(profile, tonic))[0, 1])
            if r > best_r:
                best_name, best_r = f"{NAMES[tonic]} {mode}", r
    return (best_name if best_r >= MIN_CORRELATION else "unknown"), best_r
