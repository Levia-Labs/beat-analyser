import type { Row } from "./api";

// The analysis's track names (app/events.py TRACK_ORDER), drums first, pitched last.
export const TRACK_ORDER = ["kick", "snare", "hh", "beats", "bass", "chords", "melody"];
export const SHORT: Record<string, string> = {
  kick: "KI",
  snare: "SN",
  hh: "HH",
  beats: "BT",
  bass: "BA",
  chords: "CH",
  melody: "ME",
};
export const MIN_SPAN = 2;
// Label IDs below this are MIDI pitches; 128 and up are drum and beat labels (analysis.LABELS).
export const PITCH_LIMIT = 128;

export function laneOrder(tracks: Record<string, Row[]>): string[] {
  const names = Object.keys(tracks);
  const known = TRACK_ORDER.filter((name) => names.includes(name));
  return [...known, ...names.filter((name) => !TRACK_ORDER.includes(name)).sort()];
}

export function durationOf(tracks: Record<string, Row[]>): number {
  let last = 0;
  for (const rows of Object.values(tracks)) for (const [time] of rows) last = Math.max(last, time);
  return Math.max(last, MIN_SPAN);
}

export function clampZoom(pps: number, duration: number, width: number): number {
  if (width <= 0) return pps;
  const fit = width / Math.max(duration, MIN_SPAN);
  const closest = width / MIN_SPAN;
  return Math.min(Math.max(pps, fit), Math.max(fit, closest));
}

const STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200];

export function rulerStep(pps: number): number {
  return STEPS.find((step) => step * pps >= 64) ?? 1200;
}

export function pitchRange(rows: Row[]): [number, number] | null {
  const pitches = rows.map(([, id]) => id).filter((id) => id < PITCH_LIMIT);
  if (pitches.length === 0) return null;
  return [Math.min(...pitches), Math.max(...pitches)];
}

export function pitchToY(pitch: number, range: [number, number] | null, height: number): number {
  if (!range || range[1] === range[0]) return height / 2;
  const [low, high] = range;
  return height - 2 - ((pitch - low) / (high - low)) * (height - 4);
}

export function visibleRows(rows: Row[], from: number, to: number): Row[] {
  return rows.filter(([time]) => time >= from && time <= to);
}

export function laneStats(rows: Row[]) {
  if (rows.length === 0) return { count: 0, first: null, last: null };
  const times = rows.map(([time]) => time);
  return { count: rows.length, first: Math.min(...times), last: Math.max(...times) };
}

export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}`;
}
