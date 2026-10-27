import { render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { getEvents, getProjects, renameProject } from "./api";
import App from "./App";
import { StatusChip } from "./components/StatusChip";
import Viewer, { ViewerRoute } from "./screens/Viewer";
import { useDocumentTitle } from "./useDocumentTitle";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    getConfig: vi.fn(() => new Promise(() => {})),
    getProjects: vi.fn(),
    getEvents: vi.fn(),
    renameProject: vi.fn(),
  };
});

const done = {
  name: "a",
  done: true,
  key: "A minor",
  bpm: 120,
  duration: 185,
  created: null,
  size: 5,
};
const undone = { ...done, name: "b", done: false };

beforeEach(() => {
  vi.mocked(getProjects).mockReset();
  vi.mocked(getProjects).mockResolvedValue({ items: [done, undone], total: 2 });
  vi.mocked(getEvents).mockReset();
  document.title = "Beat Analyser";
});

function at(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("masthead", () => {
  it("has the coloured wordmark and no border classes", () => {
    at("/");
    const brand = screen.getByRole("link", { name: "Beat Analyser" });
    const mark = brand.querySelector(".wordmark");
    expect(mark?.querySelector(".red")).toHaveTextContent("Beat");
    expect(mark?.querySelector(".blue")).toHaveTextContent("Analyser");
    const header = brand.closest("header");
    expect(header).toHaveClass("masthead");
    for (const element of [header, brand, document.querySelector("main")]) {
      expect(element?.className ?? "").not.toMatch(/border|rule|separator/);
    }
  });
});

describe("projects page", () => {
  it("renders the title bar outside the bordered table surface", async () => {
    at("/");
    await screen.findByRole("link", { name: "a" });
    const bar = document.querySelector(".lib-bar");
    const scroll = document.querySelector(".lib-scroll");
    expect(bar).not.toBeNull();
    expect(scroll).not.toBeNull();
    expect(scroll?.contains(bar)).toBe(false);
    expect(bar?.contains(scroll)).toBe(false);
    expect(within(bar as HTMLElement).getByRole("heading", { level: 1 })).toHaveTextContent(
      "Projects",
    );
    expect(bar?.querySelector(".count")).toHaveTextContent("2 projects");
    expect(document.querySelector(".card")).toBeNull();
  });

  it("marks Built and Not built with icon-only chips", async () => {
    at("/");
    await screen.findByRole("link", { name: "a" });
    const table = screen.getByRole("table");
    const built = within(table).getByRole("img", { name: "Built" });
    expect(built).toHaveClass("chip", "chip-ok");
    expect(built).toHaveTextContent("");
    expect(built.querySelector("svg")).toHaveAttribute("data-icon", "ok");
    const off = within(table).getByRole("img", { name: "Not built" });
    expect(off).toHaveClass("chip", "chip-error");
    expect(off.querySelector("svg")).toHaveAttribute("data-icon", "close-circle");
  });
});

describe("StatusChip", () => {
  it("labels by state", () => {
    render(<StatusChip done />);
    expect(screen.getByRole("img", { name: "Built" })).toHaveClass("chip-ok");
  });
});

const project = {
  name: "song",
  meta: { key: "A minor", bpm: 120 },
  labels: {},
  tracks: { kick: [[0.5, 128]] as [number, number][] },
};

describe("project page", () => {
  it("shows duration, created and size from the events response", async () => {
    vi.mocked(getEvents).mockResolvedValue({
      ...project,
      duration: 185,
      created: "2026-10-27T09:30:00Z",
      size: 5 * 1024 * 1024,
    });
    render(
      <MemoryRouter>
        <Viewer name="song" />
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "song" });
    expect(document.querySelector("p.sub")).toHaveTextContent("3:05 · 2026-10-27 · 5.0 MB");
  });

  it("orders back link, heading, meta row, toolbar and Timeline", async () => {
    vi.mocked(getEvents).mockResolvedValue(project);
    render(
      <MemoryRouter>
        <Viewer name="song" />
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "song" });
    const page = document.querySelector("section.page.map-page") as HTMLElement;
    const parts = [
      page.querySelector("a.back"),
      page.querySelector(".maphead h1"),
      page.querySelector(".metarow"),
      page.querySelector(".map-toolbar"),
      page.querySelector("h2.group-title"),
    ];
    for (const part of parts) expect(part).not.toBeNull();
    for (let i = 1; i < parts.length; i++) {
      const before = parts[i - 1] as Element;
      const after = parts[i] as Element;
      expect(before.compareDocumentPosition(after) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
    expect(parts[0]).toHaveTextContent("Projects");
    expect(parts[4]).toHaveTextContent("Timeline");
    const row = page.querySelector(".metarow") as HTMLElement;
    expect(within(row).getByRole("img", { name: "Built" })).toBeInTheDocument();
    expect(row.querySelector(".tempo")).toHaveTextContent("120 BPM");
    expect(page.querySelector(".map-toolbar .spacer")).not.toBeNull();
    expect(page.querySelector(".lib-scroll")).not.toBeNull();
    expect(page.querySelector(".card")).toBeNull();
  });
});

function Title({ title }: { title: string }) {
  useDocumentTitle(title);
  return null;
}

describe("document title", () => {
  it("is set, updated and restored", () => {
    const view = render(<Title title="song · Beat Analyser" />);
    expect(document.title).toBe("song · Beat Analyser");
    view.rerender(<Title title="tune · Beat Analyser" />);
    expect(document.title).toBe("tune · Beat Analyser");
    view.unmount();
    expect(document.title).toBe("Beat Analyser");
  });

  it("names the project on its page and resets on leaving", async () => {
    vi.mocked(getEvents).mockResolvedValue(project);
    const view = render(
      <MemoryRouter initialEntries={["/p/song"]}>
        <Routes>
          <Route path="/p/:name" element={<Viewer name="song" />} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "song" });
    expect(document.title).toBe("song · Beat Analyser");
    view.unmount();
    expect(document.title).toBe("Beat Analyser");
  });

  it("follows a rename", async () => {
    vi.mocked(getEvents).mockImplementation(async (name) => ({ ...project, name }));
    vi.mocked(renameProject).mockResolvedValue({ name: "tune" });
    render(
      <MemoryRouter initialEntries={["/p/song"]}>
        <Routes>
          <Route path="/p/:name" element={<ViewerRoute />} />
        </Routes>
      </MemoryRouter>,
    );
    await userEvent.click(await screen.findByRole("button", { name: "Rename" }));
    const field = screen.getByLabelText("New name");
    await userEvent.clear(field);
    await userEvent.type(field, "tune");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await vi.waitFor(() => expect(document.title).toBe("tune · Beat Analyser"));
  });

  it("is plain Beat Analyser on the list", async () => {
    at("/");
    await screen.findByRole("link", { name: "a" });
    expect(document.title).toBe("Beat Analyser");
  });
});
