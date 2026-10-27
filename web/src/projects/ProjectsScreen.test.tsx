import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, deleteProject, getProjects, renameProject, type ProjectItem } from "../api";
import ProjectsScreen from "./ProjectsScreen";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getProjects: vi.fn(), deleteProject: vi.fn(), renameProject: vi.fn() };
});

const list = vi.mocked(getProjects);
const remove = vi.mocked(deleteProject);
const rename = vi.mocked(renameProject);

function item(name: string, extra: Partial<ProjectItem> = {}): ProjectItem {
  return {
    name,
    done: true,
    key: "A minor",
    bpm: 120,
    duration: 185,
    created: "2026-10-01T12:00:00+00:00",
    size: 2_500_000,
    ...extra,
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

function show(path = "/") {
  return render(
    <MemoryRouter initialEntries={[path]}>
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
  remove.mockReset();
  rename.mockReset();
  list.mockResolvedValue({ items: [item("a"), item("b", { done: false })], total: 2 });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Projects table", () => {
  it("lists rows with the total, links, ZIP and a not-built status", async () => {
    show();
    expect(await screen.findByRole("link", { name: "a" })).toHaveAttribute("href", "/p/a");
    expect(screen.getByText("2 projects")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Download ZIP of a" })).toHaveAttribute(
      "href",
      "/download/a",
    );
    expect(screen.queryByRole("link", { name: "Download ZIP of b" })).toBeNull();
    expect(within(screen.getByRole("table")).getByText("Not built")).toBeInTheDocument();
    expect(screen.getAllByText("3:05").length).toBeGreaterThan(0);
    expect(lastParams().get("sort")).toBe("created");
  });

  it("reads filters from the URL into the request and the controls", async () => {
    show("/?q=so&key=minor&status=built&bpm_min=90&bpm_max=130&sort=bpm&order=asc");
    await screen.findByRole("link", { name: "a" });
    const params = lastParams();
    expect(params.get("q")).toBe("so");
    expect(params.get("key")).toBe("minor");
    expect(params.get("status")).toBe("built");
    expect(params.get("bpm_min")).toBe("90");
    expect(params.get("bpm_max")).toBe("130");
    expect(params.get("sort")).toBe("bpm");
    expect(params.get("order")).toBe("asc");
    expect(screen.getByRole("searchbox", { name: "Search by name" })).toHaveValue("so");
  });

  it("a header click sorts and writes the URL; a second click flips the order", async () => {
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("button", { name: "Sort by Name" }));
    expect(screen.getByTestId("url")).toHaveTextContent("sort=name");
    expect(lastParams().get("sort")).toBe("name");
    expect(lastParams().get("order")).toBe("asc");
    await userEvent.click(screen.getByRole("button", { name: "Sort by Name" }));
    // desc is the default order, so the URL drops it.
    expect(screen.getByTestId("url")).toHaveTextContent(/^\?sort=name$/);
    expect(lastParams().get("order")).toBe("desc");
    expect(screen.getByRole("columnheader", { name: /Name/ })).toHaveAttribute(
      "aria-sort",
      "descending",
    );
  });

  it("debounces the name search by 250 ms", async () => {
    show();
    await screen.findByRole("link", { name: "a" });
    vi.useFakeTimers();
    const calls = list.mock.calls.length;
    fireEvent.change(screen.getByRole("searchbox", { name: "Search by name" }), {
      target: { value: "ab" },
    });
    await act(() => vi.advanceTimersByTimeAsync(249));
    expect(list.mock.calls.length).toBe(calls);
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(list.mock.calls.length).toBeGreaterThan(calls);
    expect(lastParams().get("q")).toBe("ab");
    expect(screen.getByTestId("url")).toHaveTextContent("q=ab");
  });

  it("selects, filters by key and status, and Clear resets them", async () => {
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Key" }), "A minor");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Status" }), "built");
    expect(lastParams().get("key")).toBe("A minor");
    expect(lastParams().get("status")).toBe("built");
    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByTestId("url")).toBeEmptyDOMElement();
    expect(lastParams().has("key")).toBe(false);
  });

  it("loads the next 50 when the sentinel shows, and stops after the last page", async () => {
    list.mockResolvedValueOnce({ items: many(50), total: 70 });
    list.mockResolvedValueOnce({ items: many(20, 50), total: 70 });
    show();
    await screen.findByRole("link", { name: "song-0" });
    expect(lastParams().get("offset")).toBe("0");
    await reveal();
    expect(await screen.findByRole("link", { name: "song-69" })).toBeInTheDocument();
    expect(list).toHaveBeenCalledTimes(2);
    expect(lastParams().get("offset")).toBe("50");
    expect(lastParams().get("limit")).toBe("50");
    await reveal();
    expect(list).toHaveBeenCalledTimes(2);
  });

  it("says so when nothing exists, and when filters match nothing", async () => {
    list.mockResolvedValue({ items: [], total: 0 });
    const { unmount } = show();
    expect(await screen.findByText("No projects yet.")).toBeInTheDocument();
    unmount();
    show("/?q=zzz");
    expect(await screen.findByText("No projects match these filters.")).toBeInTheDocument();
  });

  it("shows a load error in a banner", async () => {
    list.mockRejectedValue(new ApiError("Server unreachable", 0));
    show();
    expect(await screen.findByRole("alert")).toHaveTextContent("Server unreachable");
  });

  it("renames a row inline and reloads", async () => {
    rename.mockResolvedValue({ name: "tune" });
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("button", { name: "Rename a" }));
    const field = screen.getByRole("textbox", { name: "New name" });
    await userEvent.clear(field);
    await userEvent.type(field, "tune");
    const calls = list.mock.calls.length;
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(rename).toHaveBeenCalledWith("a", "tune");
    await vi.waitFor(() => expect(list.mock.calls.length).toBeGreaterThan(calls));
    expect(screen.queryByRole("textbox", { name: "New name" })).toBeNull();
  });

  it("shows a rename failure in a banner", async () => {
    rename.mockRejectedValue(new ApiError("Name taken", 409));
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("button", { name: "Rename a" }));
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Name taken");
  });

  it("deletes one row after a confirm and reloads", async () => {
    remove.mockResolvedValue(undefined);
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("button", { name: "Delete a" }));
    const calls = list.mock.calls.length;
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Delete" }),
    );
    expect(remove).toHaveBeenCalledExactlyOnceWith("a");
    await vi.waitFor(() => expect(list.mock.calls.length).toBeGreaterThan(calls));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("bulk delete: one confirm, one DELETE per name in order, failures in a banner", async () => {
    list.mockResolvedValue({ items: [item("a"), item("b"), item("c")], total: 3 });
    remove.mockImplementation((name) =>
      name === "b" ? Promise.reject(new ApiError("Project busy", 409)) : Promise.resolve(),
    );
    show();
    await screen.findByRole("link", { name: "a" });
    expect(screen.queryByRole("button", { name: /Delete selected/ })).toBeNull();
    await userEvent.click(screen.getByRole("checkbox", { name: "Select a" }));
    await userEvent.click(screen.getByRole("checkbox", { name: "Select b" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete selected (2)" }));
    expect(remove).not.toHaveBeenCalled();
    const calls = list.mock.calls.length;
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Delete" }),
    );
    expect(remove.mock.calls).toEqual([["a"], ["b"]]);
    const banner = await screen.findByRole("alert");
    expect(banner).toHaveTextContent("b: Project busy");
    expect(banner).not.toHaveTextContent("a:");
    await vi.waitFor(() => expect(list.mock.calls.length).toBeGreaterThan(calls));
  });

  it("select-all picks every loaded row", async () => {
    list.mockResolvedValue({ items: [item("a"), item("b")], total: 2 });
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("checkbox", { name: "Select all loaded" }));
    expect(screen.getByRole("button", { name: "Delete selected (2)" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("checkbox", { name: "Select all loaded" }));
    expect(screen.queryByRole("button", { name: /Delete selected/ })).toBeNull();
  });

  it("the phone Filters button toggles the filter group by class", async () => {
    show();
    await screen.findByRole("link", { name: "a" });
    const button = screen.getByRole("button", { name: "Filters" });
    const group = screen.getByRole("group", { name: "Filters" });
    expect(group).not.toHaveClass("open");
    expect(button).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(button);
    expect(group).toHaveClass("open");
    expect(button).toHaveAttribute("aria-expanded", "true");
    await userEvent.click(button);
    expect(group).not.toHaveClass("open");
  });

  it("the phone sort select changes the sort", async () => {
    show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Sort" }), "bpm");
    expect(lastParams().get("sort")).toBe("bpm");
  });

  it("every button carries an icon", async () => {
    const { container } = show();
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("checkbox", { name: "Select a" }));
    for (const button of container.querySelectorAll("button")) {
      expect(button.querySelector("svg"), button.outerHTML).not.toBeNull();
    }
  });
});
