import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, getEvents } from "../api";
import Viewer from "./Viewer";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getEvents: vi.fn() };
});
const events = vi.mocked(getEvents);

beforeEach(() => {
  events.mockReset();
});

function show(name: string) {
  render(
    <MemoryRouter>
      <Viewer name={name} />
    </MemoryRouter>,
  );
}

describe("Viewer", () => {
  it("shows the project's facts, ZIP link and lanes", async () => {
    events.mockResolvedValue({
      name: "song",
      meta: { key: "A minor", bpm: 120 },
      labels: {},
      tracks: { kick: [[0.5, 128]] },
    });
    show("song");
    expect(await screen.findByRole("heading", { name: "song" })).toBeInTheDocument();
    expect(screen.getByText(/A minor · 120 BPM/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "ZIP" })).toHaveAttribute("href", "/download/song");
    expect(screen.getByRole("button", { name: "kick, 1 events" })).toBeInTheDocument();
  });

  it("says when a project is not found or not analysed yet", async () => {
    events.mockRejectedValue(new ApiError("Project not found", 404));
    show("nope");
    expect(await screen.findByText("No analysed project called nope")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute("href", "/");
  });

  it("shows other errors as a banner", async () => {
    events.mockRejectedValue(new ApiError("Server unreachable", 0));
    show("song");
    expect(await screen.findByRole("alert")).toHaveTextContent("Server unreachable");
  });
});
