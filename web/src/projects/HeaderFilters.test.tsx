import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getConfig, getProjects } from "../api";
import ProjectsScreen from "./ProjectsScreen";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getProjects: vi.fn(), getConfig: vi.fn() };
});

const list = vi.mocked(getProjects);

function Probe() {
  const location = useLocation();
  return <output data-testid="url">{location.search}</output>;
}

async function show(path = "/") {
  const view = render(
    <MemoryRouter initialEntries={[path]}>
      <ProjectsScreen />
      <Probe />
    </MemoryRouter>,
  );
  await screen.findByRole("link", { name: "a" });
  return view;
}

const url = () => screen.getByTestId("url");
const lastParams = () => list.mock.calls.at(-1)![0];
// hidden: an open modal popover marks everything outside it aria-hidden.
const trigger = (name: string) => screen.getByRole("button", { name, hidden: true });

beforeEach(() => {
  list.mockReset();
  vi.mocked(getConfig).mockReset();
  list.mockResolvedValue({
    items: [
      { name: "a", done: true, key: "A minor", bpm: 120, duration: 185, created: null, size: 5 },
    ],
    total: 7,
  });
});

describe("header filter popovers", () => {
  it("each column header has a sort button, a filter trigger and no trigger on Created or Size", async () => {
    await show();
    for (const name of ["Name", "Key", "BPM", "Duration", "Status"]) {
      expect(trigger(`Filter ${name}`)).toBeInTheDocument();
    }
    expect(screen.queryByRole("button", { name: "Filter Created" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Filter Size" })).toBeNull();
    expect(screen.getByRole("button", { name: "Sort by Created" })).toBeInTheDocument();
  });

  it("the sort button sets aria-sort on its column header", async () => {
    await show();
    const head = () => screen.getByRole("button", { name: "Sort by BPM" }).closest("th");
    expect(head()).toHaveAttribute("aria-sort", "none");
    expect(head()).toHaveAttribute("scope", "col");
    await userEvent.click(screen.getByRole("button", { name: "Sort by BPM" }));
    expect(head()).toHaveAttribute("aria-sort", "descending");
    await userEvent.click(screen.getByRole("button", { name: "Sort by BPM" }));
    expect(head()).toHaveAttribute("aria-sort", "ascending");
  });

  it("the Name filter moves focus into the field, writes q after the debounce, shows the badge", async () => {
    await show();
    await userEvent.click(trigger("Filter Name"));
    const box = await screen.findByRole("searchbox", { name: "Search by name" });
    expect(box).toHaveFocus();
    await userEvent.type(box, "ab");
    await vi.waitFor(() => expect(url()).toHaveTextContent("q=ab"));
    expect(lastParams().get("q")).toBe("ab");
    expect(trigger("Filter Name, 1 active")).toBeInTheDocument();
  });

  it("Escape closes the popover and returns focus to its trigger", async () => {
    await show();
    await userEvent.click(trigger("Filter Name"));
    await screen.findByRole("dialog", { name: "Name filter" });
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Name filter" })).toBeNull();
    expect(trigger("Filter Name")).toHaveFocus();
  });

  it("the close button closes the panel", async () => {
    await show();
    await userEvent.click(trigger("Filter Key"));
    await userEvent.click(await screen.findByRole("button", { name: "Close filter" }));
    expect(screen.queryByRole("dialog", { name: "Key filter" })).toBeNull();
  });

  it("the Key checklist builds a comma list in checklist order, and unchecking removes", async () => {
    await show();
    await userEvent.click(trigger("Filter Key"));
    const panel = await screen.findByRole("dialog", { name: "Key filter" });
    await userEvent.click(within(panel).getByRole("checkbox", { name: "A minor" }));
    await userEvent.click(within(panel).getByRole("checkbox", { name: "Minor" }));
    await userEvent.click(within(panel).getByRole("checkbox", { name: "C major" }));
    expect(lastParams().get("key")).toBe("minor,C major,A minor");
    expect(new URLSearchParams(url().textContent ?? "").get("key")).toBe("minor,C major,A minor");
    await userEvent.click(within(panel).getByRole("checkbox", { name: "Minor" }));
    expect(lastParams().get("key")).toBe("C major,A minor");
    expect(trigger("Filter Key, 2 active")).toBeInTheDocument();
  });

  it("the Status checklist: one choice filters, both or none mean any", async () => {
    await show();
    await userEvent.click(trigger("Filter Status"));
    const panel = await screen.findByRole("dialog", { name: "Status filter" });
    await userEvent.click(within(panel).getByRole("checkbox", { name: "Built" }));
    expect(lastParams().get("status")).toBe("built");
    await userEvent.click(within(panel).getByRole("checkbox", { name: "Not built" }));
    expect(lastParams().has("status")).toBe(false);
    await userEvent.click(within(panel).getByRole("checkbox", { name: "Not built" }));
    expect(lastParams().get("status")).toBe("pending");
  });

  it("BPM and Duration ranges write their parameters, replacing history entries", async () => {
    await show();
    await userEvent.click(trigger("Filter BPM"));
    await userEvent.type(await screen.findByRole("textbox", { name: "BPM min" }), "90");
    await userEvent.type(screen.getByRole("textbox", { name: "BPM max" }), "140");
    expect(lastParams().get("bpm_min")).toBe("90");
    expect(lastParams().get("bpm_max")).toBe("140");
    await userEvent.keyboard("{Escape}");
    await userEvent.click(trigger("Filter Duration"));
    await userEvent.type(await screen.findByRole("textbox", { name: "Duration min (s)" }), "60");
    expect(lastParams().get("dur_min")).toBe("60");
    expect(trigger("Filter BPM, 1 active")).toBeInTheDocument();
    expect(trigger("Filter Duration, 1 active")).toBeInTheDocument();
  });

  it("a range refuses min above max and says so, keeping the URL valid", async () => {
    await show("/?bpm_min=100");
    await userEvent.click(trigger("Filter BPM, 1 active"));
    const max = await screen.findByRole("textbox", { name: "BPM max" });
    await userEvent.type(max, "50");
    expect(await screen.findByRole("alert")).toHaveTextContent("Minimum is above the maximum");
    expect(max).toBeInvalid();
    expect(url()).toHaveTextContent("bpm_min=100");
    expect(url()).not.toHaveTextContent("bpm_max");
    await userEvent.clear(max);
    await userEvent.type(max, "150");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(lastParams().get("bpm_max")).toBe("150");
  });

  it("a range refuses text that is not a number", async () => {
    await show();
    await userEvent.click(trigger("Filter Duration"));
    const min = await screen.findByRole("textbox", { name: "Duration min (s)" });
    await userEvent.type(min, "x");
    expect(min).toBeInvalid();
    expect(url()).toBeEmptyDOMElement();
  });

  it("per-column Clear removes only that column's filter", async () => {
    await show("/?q=zz&bpm_min=90&status=built");
    await userEvent.click(trigger("Filter BPM, 1 active"));
    await userEvent.click(await screen.findByRole("button", { name: "Clear BPM filter" }));
    expect(url()).not.toHaveTextContent("bpm_min");
    expect(url()).toHaveTextContent("q=zz");
    expect(url()).toHaveTextContent("status=built");
  });

  it("the per-column Clear is disabled while the column has no filter", async () => {
    await show();
    await userEvent.click(trigger("Filter BPM"));
    expect(await screen.findByRole("button", { name: "Clear BPM filter" })).toBeDisabled();
  });

  it("Clear filters (n) shows only with filters, counts them all, and keeps the sort", async () => {
    await show();
    expect(screen.queryByRole("button", { name: /Clear filters/ })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Sort by Name" }));
    await userEvent.click(trigger("Filter Key"));
    await userEvent.click(await screen.findByRole("checkbox", { name: "A minor" }));
    await userEvent.click(await screen.findByRole("checkbox", { name: "C major" }));
    await userEvent.keyboard("{Escape}");
    await userEvent.click(trigger("Filter Status"));
    await userEvent.click(await screen.findByRole("checkbox", { name: "Built" }));
    await userEvent.keyboard("{Escape}");
    await userEvent.click(screen.getByRole("button", { name: "Clear filters (3)" }));
    expect(url()).toHaveTextContent(/^\?sort=name&order=asc$/);
    expect(screen.queryByRole("button", { name: /Clear filters/ })).toBeNull();
  });
});

describe("phone filter sheet", () => {
  it("groups every column, Clear all is disabled at zero, Show N closes the sheet", async () => {
    await show();
    await userEvent.click(screen.getByRole("button", { name: /^Filters/ }));
    const sheet = await screen.findByRole("dialog", { name: "Filters" });
    for (const group of ["Name", "Key", "BPM", "Duration", "Status"]) {
      expect(within(sheet).getByRole("heading", { name: group })).toBeInTheDocument();
    }
    expect(within(sheet).getByRole("button", { name: "Clear all" })).toBeDisabled();
    await userEvent.click(within(sheet).getByRole("checkbox", { name: "Built" }));
    expect(lastParams().get("status")).toBe("built");
    expect(within(sheet).getByRole("button", { name: "Clear all" })).toBeEnabled();
    await userEvent.click(within(sheet).getByRole("button", { name: "Show 7 projects" }));
    expect(screen.queryByRole("dialog", { name: "Filters" })).toBeNull();
    expect(screen.getByRole("button", { name: "Filters, 1 active" })).toBeInTheDocument();
  });

  it("Clear all empties every filter; the sort select and order toggle change the sort", async () => {
    await show("/?q=zz&bpm_max=90");
    await userEvent.click(screen.getByRole("button", { name: /^Filters/ }));
    const sheet = await screen.findByRole("dialog", { name: "Filters" });
    await userEvent.selectOptions(within(sheet).getByRole("combobox", { name: "Sort" }), "bpm");
    expect(lastParams().get("sort")).toBe("bpm");
    await userEvent.click(within(sheet).getByRole("button", { name: /Descending, reverse/ }));
    expect(lastParams().get("order")).toBe("asc");
    await userEvent.click(within(sheet).getByRole("button", { name: "Clear all" }));
    expect(url()).toHaveTextContent(/^\?sort=bpm&order=asc$/);
  });

  it("every button in the sheet and the popovers carries an icon", async () => {
    await show();
    await userEvent.click(screen.getByRole("button", { name: /^Filters/ }));
    const sheet = await screen.findByRole("dialog", { name: "Filters" });
    for (const button of sheet.querySelectorAll("button")) {
      expect(button.querySelector("svg"), button.outerHTML).not.toBeNull();
    }
  });
});
