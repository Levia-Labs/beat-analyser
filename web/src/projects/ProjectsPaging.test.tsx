import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, getProjects, renameProject, type ProjectItem } from "../api";
import ProjectsScreen from "./ProjectsScreen";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getProjects: vi.fn(), deleteProject: vi.fn(), renameProject: vi.fn() };
});

const list = vi.mocked(getProjects);
const rename = vi.mocked(renameProject);

function item(name: string): ProjectItem {
  return {
    name,
    done: true,
    key: "A minor",
    bpm: 120,
    duration: 185,
    created: "2026-10-01T12:00:00+00:00",
    size: 2_500_000,
  };
}

function many(count: number, from = 0) {
  return Array.from({ length: count }, (_, index) => item(`song-${from + index}`));
}

class FakeObserver {
  static all: FakeObserver[] = [];
  disconnected = false;
  constructor(readonly callback: IntersectionObserverCallback) {
    FakeObserver.all.push(this);
  }
  observe() {}
  unobserve() {}
  disconnect() {
    this.disconnected = true;
  }
}

async function reveal() {
  await act(async () => {
    for (const observer of FakeObserver.all.filter((o) => !o.disconnected)) {
      observer.callback(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        observer as unknown as IntersectionObserver,
      );
    }
  });
}

function Probe() {
  const location = useLocation();
  return <output data-testid="url">{location.search}</output>;
}

function show() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <ProjectsScreen />
      <Probe />
    </MemoryRouter>,
  );
}

const lastParams = () => list.mock.calls.at(-1)![0];

beforeEach(() => {
  FakeObserver.all = [];
  vi.stubGlobal("IntersectionObserver", FakeObserver);
  list.mockReset();
  rename.mockReset();
  list.mockResolvedValue({ items: [item("a"), item("b")], total: 2 });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Projects paging and reload", () => {
  it("an overlapping page adds no duplicate rows and keeps loading while short", async () => {
    list.mockResolvedValueOnce({ items: many(50), total: 70 });
    // A row inserted above shifted the list: this page repeats song-45..49.
    list.mockResolvedValueOnce({ items: many(20, 45), total: 70 });
    list.mockResolvedValueOnce({ items: many(5, 65), total: 70 });
    show();
    await screen.findByRole("link", { name: "song-0" });
    await reveal();
    expect(await screen.findByRole("link", { name: "song-69" })).toBeInTheDocument();
    expect(list).toHaveBeenCalledTimes(3);
    expect(lastParams().get("offset")).toBe("65");
    const names = screen.getAllByRole("link", { name: /^song-/ }).map((a) => a.textContent);
    expect(names).toHaveLength(70);
    expect(new Set(names).size).toBe(70);
  });

  it("a failed page shows the error with Retry, which asks for the same page", async () => {
    list.mockResolvedValueOnce({ items: many(50), total: 70 });
    list.mockRejectedValueOnce(new ApiError("Server unreachable", 0));
    list.mockResolvedValueOnce({ items: many(20, 50), total: 70 });
    show();
    await screen.findByRole("link", { name: "song-0" });
    await reveal();
    expect(await screen.findByRole("alert")).toHaveTextContent("Server unreachable");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("link", { name: "song-69" })).toBeInTheDocument();
    expect(lastParams().get("offset")).toBe("50");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("reloads in place: rows stay on screen until the new data arrives", async () => {
    rename.mockResolvedValue({ name: "tune" });
    show();
    await screen.findByRole("link", { name: "a" });
    let finish: (value: { items: ProjectItem[]; total: number }) => void = () => {};
    list.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    await userEvent.click(screen.getByRole("button", { name: "Rename a" }));
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await vi.waitFor(() => expect(lastParams().get("limit")).toBe("50"));
    expect(lastParams().get("offset")).toBe("0");
    expect(screen.getByRole("link", { name: "b" })).toBeInTheDocument();
    expect(screen.queryByText("Loading…")).toBeNull();
    await act(async () => finish({ items: [item("tune"), item("b")], total: 2 }));
    expect(await screen.findByRole("link", { name: "tune" })).toBeInTheDocument();
  });

  it("Clear inside the debounce window drops the typed text for good", async () => {
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("button", { name: "Filter Name" }));
    const box = await screen.findByRole("searchbox", { name: "Search by name" });
    vi.useFakeTimers();
    fireEvent.change(box, { target: { value: "ab" } });
    await act(() => vi.advanceTimersByTimeAsync(100));
    fireEvent.click(screen.getByRole("button", { name: "Clear Name filter" }));
    await act(() => vi.advanceTimersByTimeAsync(600));
    expect(box).toHaveValue("");
    expect(screen.getByTestId("url")).toBeEmptyDOMElement();
  });
});
