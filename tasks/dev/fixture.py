#!/usr/bin/env python3
"""Write a synthetic test clip with known events. Standard library only.

Kick (55 Hz, decaying) on every beat, hi-hat (noise burst) on every off-beat, and an A4 tone
(MIDI 69) from second 1 to 2 - so a correct analysis is known in advance: kicks at
0, 60/bpm, ..., hh at half a beat later, note_69 (ID 69) near 1.0 s in chords.csv.

Usage:
  python3 tasks/dev/fixture.py OUT.wav [--seconds 4] [--bpm 120] [--mp3]

--mp3 also writes OUT.mp3 through ffmpeg. Names under uploads/ should start with `fixture-`,
which `task uploads:fixtures:clean` is allowed to delete.
"""

import argparse
import math
import random
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

RATE = 44100


def samples(seconds: float, bpm: float) -> list[float]:
    beat = 60.0 / bpm
    noise = random.Random(0)
    out = []
    for i in range(int(seconds * RATE)):
        t = i / RATE
        since_beat = t % beat
        since_offbeat = (t - beat / 2) % beat
        kick = math.sin(2 * math.pi * 55 * t) * math.exp(-30 * since_beat)
        hat = noise.uniform(-1, 1) * math.exp(-80 * since_offbeat) * 0.3
        tone = 0.3 * math.sin(2 * math.pi * 440 * t) if 1.0 <= t < 2.0 else 0.0
        out.append(0.3 * (kick + hat + tone))
    return out


def write_wav(path: Path, data: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in data)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--mp3", action="store_true", help="also write an MP3 via ffmpeg")
    args = parser.parse_args()

    write_wav(args.out, samples(args.seconds, args.bpm))
    print(args.out)
    if args.mp3:
        if not shutil.which("ffmpeg"):
            print("ffmpeg not on PATH - MP3 skipped", file=sys.stderr)
            return 1
        mp3 = args.out.with_suffix(".mp3")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(args.out), str(mp3)],
            check=True,
        )
        print(mp3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
