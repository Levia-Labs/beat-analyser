import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Events } from "../api";
import { ZoomInIcon, ZoomOutIcon } from "./icons";
import {
  clampZoom,
  durationOf,
  formatTime,
  laneOrder,
  laneStats,
  pitchRange,
  pitchToY,
  rulerStep,
  SHORT,
  visibleRows,
} from "../timeline";

const RULER_H = 20;
const COLOURS: Record<string, string> = {
  kick: "--red",
  snare: "--amber",
  hh: "--blue",
  beats: "--green",
  bass: "--ink",
  chords: "--muted",
  melody: "--ink",
};

function laneHeight(): number {
  return window.matchMedia?.("(max-width: 719px)").matches ? 24 : 28;
}

// One viewport-wide canvas, redrawn for the visible window: a song at full zoom would be far
// wider than the largest canvas a browser allows. A spacer gives the scrollbar its length.
export default function Timeline({ events }: { events: Events }) {
  const lanes = laneOrder(events.tracks);
  const duration = durationOf(events.tracks);
  const scroller = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [left, setLeft] = useState(0);
  const [zoom, setZoom] = useState(0);
  const [picked, setPicked] = useState<string | null>(null);
  const spacer = useRef<HTMLDivElement>(null);
  const pendingLeft = useRef<number | null>(null);
  const pps = clampZoom(zoom || (width ? width / duration : 1), duration, width);
  const view = useRef({ pps, left, width });

  function zoomAt(factor: number, anchor: number) {
    const { pps: current, left: from, width: visible } = view.current;
    const next = clampZoom(current * factor, duration, visible);
    const time = (from + anchor) / current;
    pendingLeft.current = Math.max(0, time * next - anchor);
    setZoom(next);
  }
  const zoomRef = useRef(zoomAt);

  // Refs are written after render, never during it (react-hooks rule); the native listeners
  // read them when an event arrives.
  useLayoutEffect(() => {
    view.current = { pps, left, width };
    zoomRef.current = zoomAt;
  });

  // The spacer's width goes through the DOM: the CSP refuses style="" in HTML, not this.
  useEffect(() => {
    if (spacer.current) spacer.current.style.width = `${duration * pps}px`;
  }, [duration, pps]);

  useEffect(() => {
    const element = scroller.current;
    if (!element) return;
    const measure = () => setWidth(element.clientWidth);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Native listeners: React's wheel and touch listeners are passive, and only an active
  // listener can stop ctrl+wheel and pinch from zooming the whole page.
  useEffect(() => {
    const element = scroller.current;
    if (!element) return;
    let pinch = 0;
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const anchor = event.clientX - element.getBoundingClientRect().left;
      zoomRef.current(event.deltaY < 0 ? 1.25 : 0.8, anchor);
    };
    const onTouchMove = (event: TouchEvent) => {
      const first = event.touches.item(0);
      const second = event.touches.item(1);
      if (event.touches.length !== 2 || !first || !second) return;
      event.preventDefault();
      const now = Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
      if (pinch) {
        const middle = (first.clientX + second.clientX) / 2;
        zoomRef.current(now / pinch, middle - element.getBoundingClientRect().left);
      }
      pinch = now;
    };
    const onTouchEnd = () => {
      pinch = 0;
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    element.addEventListener("touchmove", onTouchMove, { passive: false });
    element.addEventListener("touchend", onTouchEnd);
    return () => {
      element.removeEventListener("wheel", onWheel);
      element.removeEventListener("touchmove", onTouchMove);
      element.removeEventListener("touchend", onTouchEnd);
    };
  }, []);

  useEffect(() => {
    if (pendingLeft.current !== null && scroller.current) {
      scroller.current.scrollLeft = pendingLeft.current;
      pendingLeft.current = null;
    }
  }, [pps]);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext?.("2d");
    if (!element || !context || width === 0) return;
    const lane = laneHeight();
    const height = RULER_H + lanes.length * lane;
    const ratio = window.devicePixelRatio || 1;
    element.width = Math.round(width * ratio);
    element.height = Math.round(height * ratio);
    element.style.width = `${width}px`;
    element.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const css = getComputedStyle(element);
    const colour = (name: string) => css.getPropertyValue(name).trim() || "#ece9f7";
    context.clearRect(0, 0, width, height);

    const from = left / pps;
    const to = (left + width) / pps;
    const step = rulerStep(pps);
    context.fillStyle = colour("--muted");
    context.font = "11px sans-serif";
    for (let time = Math.ceil(from / step) * step; time <= to; time += step) {
      const x = time * pps - left;
      context.fillRect(x, RULER_H - 6, 1, 6);
      context.fillText(formatTime(time).replace(/\.0$/, ""), x + 3, 11);
    }

    lanes.forEach((track, index) => {
      const top = RULER_H + index * lane;
      context.fillStyle = colour("--line");
      context.fillRect(0, top, width, 1);
      const rows = events.tracks[track] ?? [];
      const range = pitchRange(rows);
      context.fillStyle = colour(COLOURS[track] ?? "--ink");
      for (const [time, id] of visibleRows(rows, from, to)) {
        const x = time * pps - left;
        if (range && id < 128) {
          context.fillRect(x, top + pitchToY(id, range, lane) - 2, 3, 4);
        } else {
          context.fillRect(x, top + 4, 2, lane - 8);
        }
      }
    });
  }, [events, lanes, pps, left, width]);

  const stats = picked ? laneStats(events.tracks[picked] ?? []) : null;
  return (
    <section className="card timeline">
      <div className="zoom">
        <button type="button" aria-label="Zoom out" onClick={() => zoomAt(0.8, width / 2)}>
          <ZoomOutIcon />
        </button>
        <button type="button" aria-label="Zoom in" onClick={() => zoomAt(1.25, width / 2)}>
          <ZoomInIcon />
        </button>
        <span className="muted num">{formatTime(duration)}</span>
      </div>
      <div className="lanes">
        <div className="labels">
          {lanes.map((track) => {
            const count = events.tracks[track]?.length ?? 0;
            return (
              <button
                key={track}
                type="button"
                data-track={track}
                className={picked === track ? "lane on" : "lane"}
                aria-label={`${track}, ${count} events`}
                onClick={() => setPicked(picked === track ? null : track)}
              >
                <span className="long">{track}</span>
                <span className="short">{SHORT[track] ?? track.slice(0, 2).toUpperCase()}</span>
              </button>
            );
          })}
        </div>
        <div
          ref={scroller}
          className="scroller"
          data-testid="timeline-scroller"
          onScroll={(event) => setLeft(event.currentTarget.scrollLeft)}
        >
          <div className="spacer" ref={spacer}>
            <canvas
              ref={canvas}
              role="img"
              aria-label={`Timeline of ${lanes.length} tracks, ${formatTime(duration)} long`}
            />
          </div>
        </div>
      </div>
      {picked && stats && (
        <p className="muted num">
          {picked}: {stats.count} events
          {stats.first !== null && stats.last !== null
            ? `, ${formatTime(stats.first)} to ${formatTime(stats.last)}`
            : ""}
        </p>
      )}
    </section>
  );
}
