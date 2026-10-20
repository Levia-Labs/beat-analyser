#!/usr/bin/env python3
"""Score beatcore's tempo, beats and onset labels against ground truth.

Two sources of truth:

  synthetic  generated here: five styles across 70-180 BPM with known kick, snare, hi-hat
             and beat times. Runs anywhere beatcore and numpy are installed.
  DIR        a folder of audio with annotations beside it, `<stem>.beats` (one beat time per
             line, extra columns ignored - the Ballroom annotation format) and optionally
             `<stem>.bpm` (one number). Audio other than WAV is decoded with ffmpeg.

Metrics, as in the MIREX tempo and beat tasks:

  acc1   tempo within 4 percent of the truth
  acc2   the same, also accepting half, double, a third and three times the tempo
  beatF  beat F-measure, a beat matching within 70 ms
  onsetF per-class onset F-measure within 50 ms (synthetic only - real sets have no labels)

Usage (inside the image, where beatcore is built - task eval:synthetic, task eval:dir):
  python /eval/eval.py synthetic [--seconds 20]
  python /eval/eval.py /data
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import beatcore  # pyright: ignore[reportMissingImports]  # in the image; on the host only with Rust

RATE = 44100
STYLES = ("four", "noisehat", "backbeat", "swing", "syncopated", "pad")
TEMPOS = (70, 90, 110, 128, 150, 180)


def _decay(t, hits, rate):
    env = np.zeros_like(t)
    for h in hits:
        mask = t >= h
        env[mask] += np.exp(-rate * (t[mask] - h))
    return env


def synth(style: str, bpm: float, seconds: float, seed: int):
    """Audio and truth: {"beats": [...], "kick": [...], "snare": [...], "hh": [...]}."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    beat = 60.0 / bpm
    beats = np.arange(0, seconds - 0.1, beat)
    eighths = np.arange(0, seconds - 0.1, beat / 2)
    kick, snare, hh = list(beats), [], list(eighths[1::2])
    if style in ("backbeat", "swing", "syncopated", "pad"):
        kick, snare = list(beats[0::2]), list(beats[1::2])
    if style == "swing":
        hh = [b + beat * 2 / 3 for b in beats if b + beat * 2 / 3 < seconds - 0.1]
    if style == "syncopated":
        kick += [b + beat * 0.75 for b in beats[1::4]]
    kick.sort()

    y = 0.6 * np.sin(2 * np.pi * 55 * t) * _decay(t, kick, 30)
    y += 0.4 * rng.normal(0, 1, len(t)) * _decay(t, snare, 25) * 0.6
    y += 0.4 * np.sin(2 * np.pi * 200 * t) * _decay(t, snare, 25)
    # noisehat: white-noise hats like task dev:fixture; they spill into the snare band.
    white = style == "noisehat"
    hat_noise = rng.normal(0, 1, len(t)) if white else np.diff(rng.normal(0, 1, len(t) + 1))
    y += 0.25 * hat_noise * _decay(t, hh, 90)
    if style == "pad":
        chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63))
        y += 0.08 * chord + 0.02 * rng.normal(0, 1, len(t))
    y = 0.8 * y / max(1e-9, np.max(np.abs(y)))
    truth = {"beats": list(beats), "kick": kick, "snare": snare, "hh": hh}
    return y.astype(np.float32), truth


def f_measure(found, truth, window):
    found, truth = sorted(found), sorted(truth)
    if not found and not truth:
        return 1.0
    if not found or not truth:
        return 0.0
    used, hits = set(), 0
    for t in truth:
        best = None
        for i, f in enumerate(found):
            if (
                i not in used
                and abs(f - t) <= window
                and (best is None or abs(f - t) < abs(found[best] - t))
            ):
                best = i
        if best is not None:
            used.add(best)
            hits += 1
    precision, recall = hits / len(found), hits / len(truth)
    return 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)


def tempo_scores(found: float, truth: float) -> tuple[bool, bool]:
    def close(x):
        return abs(found - x) <= 0.04 * x

    acc1 = close(truth)
    acc2 = any(close(truth * k) for k in (1, 2, 0.5, 3, 1 / 3))
    return acc1, acc2


def score(result, truth_bpm, truth):
    acc1, acc2 = tempo_scores(result["bpm"], truth_bpm)
    row = {
        "bpm": result["bpm"],
        "acc1": acc1,
        "acc2": acc2,
        "beatF": f_measure(result["beats"], truth["beats"], 0.07),
    }
    for label in ("kick", "snare", "hh"):
        if label in truth:
            found = [time for time, lab in result["onsets"] if lab == label]
            row[label] = f_measure(found, truth[label], 0.05)
    return row


def run_synthetic(seconds: float) -> list[tuple[str, dict]]:
    rows = []
    for i, style in enumerate(STYLES):
        for j, bpm in enumerate(TEMPOS):
            y, truth = synth(style, bpm, seconds, seed=i * 100 + j)
            result = beatcore.analyse_samples(y.tolist(), RATE)
            rows.append((f"{style}@{bpm}", score(result, bpm, truth)))
    return rows


def read_numbers(path: Path) -> list[float]:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        if parts:
            values.append(float(parts[0]))
    return values


def run_dir(folder: Path) -> list[tuple[str, dict]]:
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for beats_file in sorted(folder.rglob("*.beats")):
            audio = next(
                (p for p in beats_file.parent.glob(beats_file.stem + ".*")
                 if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")),
                None,
            )  # fmt: skip
            if audio is None:
                continue
            truth_beats = read_numbers(beats_file)
            bpm_file = beats_file.with_suffix(".bpm")
            if bpm_file.is_file():
                truth_bpm = read_numbers(bpm_file)[0]
            elif len(truth_beats) > 1:
                truth_bpm = 60.0 / float(np.median(np.diff(truth_beats)))
            else:
                continue
            wav = audio
            if audio.suffix.lower() != ".wav":
                wav = Path(scratch) / "clip.wav"
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(audio), str(wav)],
                    check=True,
                )
            result = beatcore.analyse(str(wav))
            rows.append((audio.stem, score(result, truth_bpm, {"beats": truth_beats})))
    return rows


def report(rows: list[tuple[str, dict]]) -> None:
    columns = [c for c in ("acc1", "acc2", "beatF", "kick", "snare", "hh") if c in rows[0][1]]
    print(f"{'clip':<24} {'bpm':>7} " + " ".join(f"{c:>6}" for c in columns))
    for name, row in rows:
        cells = " ".join(
            f"{('yes' if row[c] else 'NO') if isinstance(row[c], bool) else f'{row[c]:.2f}':>6}"
            for c in columns
        )
        print(f"{name:<24} {row['bpm']:7.2f} {cells}")
    print("-" * (32 + 7 * len(columns)))
    means = " ".join(f"{np.mean([float(r[c]) for _, r in rows]):6.2f}" for c in columns)
    print(f"{'mean over ' + str(len(rows)) + ' clips':<24} {'':>7} {means}")


def as_table(rows: list[tuple[str, dict]]) -> dict[str, dict[str, float]]:
    return {name: {k: float(v) for k, v in row.items()} for name, row in rows}


def diff(rows: list[tuple[str, dict]], baseline: dict, threshold: float) -> None:
    """Means against the baseline, then only the clip scores that moved by the threshold.

    The full table is 36 rows; a tuning loop needs to know what moved (ADR 0011).
    """
    current = as_table(rows)
    columns = [c for c in ("acc1", "acc2", "beatF", "kick", "snare", "hh") if c in rows[0][1]]
    cells = []
    for column in columns:
        now = float(np.mean([r[column] for r in current.values()]))
        before = [r[column] for r in baseline.values() if column in r]
        delta = f" ({now - float(np.mean(before)):+.2f})" if before else ""
        cells.append(f"{column} {now:.2f}{delta}")
    print(f"mean over {len(current)}: " + "  ".join(cells))
    if not baseline:
        print("no baseline: task eval:synthetic:baseline saves this run as one")
        return
    moved = [
        f"  {name} {column} {baseline[name][column]:.2f} > {row[column]:.2f}"
        for name, row in current.items()
        if name in baseline
        for column in columns
        if column in baseline[name] and abs(row[column] - baseline[name][column]) >= threshold
    ]
    print(f"{len(moved)} clip score(s) moved by {threshold} or more")
    print("\n".join(moved[:25]))


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("source", help="`synthetic`, or a folder of audio and .beats files")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--save", type=Path, help="write the scores here as JSON")
    parser.add_argument("--baseline", type=Path, help="print only what moved against this")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    synthetic = args.source == "synthetic"
    rows = run_synthetic(args.seconds) if synthetic else run_dir(Path(args.source))
    if not rows:
        print("no clips with annotations found", file=sys.stderr)
        return 1
    if args.save:
        args.save.write_text(json.dumps(as_table(rows), indent=1), encoding="utf-8")
    if args.baseline:
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            baseline = {}
        diff(rows, baseline, args.threshold)
    else:
        report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
