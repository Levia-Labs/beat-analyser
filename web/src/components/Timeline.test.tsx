import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Events } from "../api";
import Timeline from "./Timeline";

const events: Events = {
  name: "song",
  meta: { key: "A minor", bpm: 120 },
  labels: { "131": "beat" },
  tracks: {
    beats: [
      [0.5, 131],
      [1, 131],
      [1.5, 131],
      [2, 131],
    ],
    kick: [[0.25, 128]],
  },
};

describe("Timeline", () => {
  it("draws one labelled lane per track in lane order", () => {
    render(<Timeline events={events} />);
    const lanes = screen.getAllByRole("button", { name: /events$/ });
    expect(lanes.map((lane) => lane.dataset.track)).toEqual(["kick", "beats"]);
  });

  it("shows a lane's count and span when its label is chosen", () => {
    render(<Timeline events={events} />);
    fireEvent.click(screen.getByRole("button", { name: "beats, 4 events" }));
    expect(screen.getByText(/beats: 4 events, 0:00.5 to 0:02.0/)).toBeInTheDocument();
  });

  it("zooms with the buttons without throwing in a canvas-less DOM", () => {
    render(<Timeline events={events} />);
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(screen.getByRole("img", { name: /Timeline of 2 tracks/ })).toBeInTheDocument();
  });

  it("stops ctrl+wheel from zooming the page", () => {
    render(<Timeline events={events} />);
    const scroller = screen.getByTestId("timeline-scroller");
    const wheel = new WheelEvent("wheel", { deltaY: -100, ctrlKey: true, cancelable: true });
    scroller.dispatchEvent(wheel);
    expect(wheel.defaultPrevented).toBe(true);
  });
});
