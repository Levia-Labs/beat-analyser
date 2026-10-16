import pytest

import key


def scale(tonic_midi, steps, seconds=0.5):
    return [(i * seconds, (i + 1) * seconds, tonic_midi + s, 0.8) for i, s in enumerate(steps)]


MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11, 12, 7, 4, 0)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10, 12, 7, 3, 0)


@pytest.mark.parametrize(
    ("tonic", "steps", "expected"),
    [
        (60, MAJOR_STEPS, "C major"),
        (67, MAJOR_STEPS, "G major"),
        (57, MINOR_STEPS, "A minor"),
        (62, MINOR_STEPS, "D minor"),
    ],
)
def test_scales(tonic, steps, expected):
    assert key.estimate(scale(tonic, steps))[0] == expected


def test_too_few_pitch_classes_is_unknown():
    assert key.estimate([(0.0, 1.0, 69, 0.9), (1.0, 2.0, 81, 0.9)]) == ("unknown", 0.0)


def test_empty_is_unknown():
    assert key.estimate([])[0] == "unknown"
