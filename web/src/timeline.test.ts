import { describe, expect, it } from "vitest";
import {
  clampZoom,
  durationOf,
  formatTime,
  laneOrder,
  laneStats,
  pitchRange,
  pitchToY,
  rulerStep,
  visibleRows,
} from "./timeline";

describe("timeline maths", () => {
  it("orders lanes like the analysis, unknown tracks last", () => {
    expect(laneOrder({ melody: [], zeta: [], kick: [], beats: [] })).toEqual([
      "kick",
      "beats",
      "melody",
      "zeta",
    ]);
  });

  it("takes the latest event as the duration, at least two seconds", () => {
    expect(durationOf({ kick: [[1, 128]], bass: [[95.5, 40]] })).toBe(95.5);
    expect(durationOf({})).toBe(2);
  });

  it("clamps zoom between the whole song and two seconds per viewport", () => {
    expect(clampZoom(1, 100, 800)).toBe(8);
    expect(clampZoom(10_000, 100, 800)).toBe(400);
    expect(clampZoom(50, 100, 800)).toBe(50);
  });

  it("picks a ruler step at least 64 px wide", () => {
    expect(rulerStep(200)).toBe(0.5);
    expect(rulerStep(10)).toBe(10);
    expect(rulerStep(0.1)).toBe(1200);
  });

  it("maps pitch into the lane, lowest at the bottom", () => {
    const range = pitchRange([
      [0, 40],
      [1, 52],
      [2, 128],
    ]);
    expect(range).toEqual([40, 52]);
    expect(pitchToY(40, range, 28)).toBe(26);
    expect(pitchToY(52, range, 28)).toBe(2);
    expect(pitchToY(60, null, 28)).toBe(14);
  });

  it("keeps only the rows inside the visible window", () => {
    const rows: [number, number][] = [
      [0, 1],
      [1, 1],
      [2, 1],
      [3, 1],
    ];
    expect(visibleRows(rows, 0.5, 2.5)).toEqual([
      [1, 1],
      [2, 1],
    ]);
  });

  it("summarises a lane and formats times", () => {
    expect(
      laneStats([
        [0.5, 1],
        [61.25, 1],
      ]),
    ).toEqual({ count: 2, first: 0.5, last: 61.25 });
    expect(laneStats([])).toEqual({ count: 0, first: null, last: null });
    expect(formatTime(61.25)).toBe("1:01.3");
  });
});
