import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getProjects } from "../api";
import ProjectsScreen from "../projects/ProjectsScreen";
import AddModal from "./AddModal";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getProjects: vi.fn() };
});

const trigger = (name: string) => screen.getByRole("button", { name, hidden: true });

async function show(path = "/") {
  render(
    <MemoryRouter initialEntries={[path]}>
      <ProjectsScreen />
    </MemoryRouter>,
  );
  await screen.findByRole("link", { name: "a" });
}

beforeEach(() => {
  vi.mocked(getProjects).mockReset();
  vi.mocked(getProjects).mockResolvedValue({
    items: [
      { name: "a", done: true, key: "A minor", bpm: 120, duration: 185, created: null, size: 5 },
    ],
    total: 7,
  });
});

describe("the Add button", () => {
  it("is a btn primary whose icon and label are children of the one button", () => {
    render(
      <MemoryRouter>
        <AddModal />
      </MemoryRouter>,
    );
    const add = screen.getByRole("button", { name: "Add" });
    expect(add).toHaveClass("btn", "primary");
    expect(add.querySelector(":scope > svg.ic")).not.toBeNull();
    const label = add.querySelector(":scope > .label-hide-mobile");
    expect(label).toHaveTextContent("Add");
  });
});

// jsdom does not evaluate the media query that shows .filters-button on a phone; this checks the
// class that library.css hides on desktop.
describe("the phone Filters button", () => {
  it("carries filters-button next to btn", async () => {
    await show();
    expect(screen.getByRole("button", { name: /^Filters/ })).toHaveClass("btn", "filters-button");
  });
});

describe("clear icons", () => {
  it("every per-column clear is an icon-only btn with the filter-x icon", async () => {
    await show("/?q=zz&bpm_min=90&status=built&key=minor&dur_min=3");
    for (const column of ["Name", "Key", "BPM", "Duration", "Status"]) {
      await userEvent.click(trigger(`Filter ${column}, 1 active`));
      const panel = await screen.findByRole("dialog", { name: `${column} filter` });
      const clear = within(panel).getByRole("button", { name: `Clear ${column} filter` });
      expect(clear).toHaveClass("btn", "icon");
      expect(clear.querySelector("svg.ic")).toHaveAttribute("data-icon", "filter-x");
      await userEvent.keyboard("{Escape}");
    }
  });

  it("Clear all and Clear filters (n) use the same icon", async () => {
    await show("/?status=built");
    expect(
      screen.getByRole("button", { name: "Clear filters (1)" }).querySelector("svg"),
    ).toHaveAttribute("data-icon", "filter-x");
    await userEvent.click(screen.getByRole("button", { name: /^Filters/ }));
    const sheet = await screen.findByRole("dialog", { name: "Filters" });
    const all = within(sheet).getByRole("button", { name: "Clear all" });
    expect(all).toHaveTextContent("");
    expect(all.querySelector("svg")).toHaveAttribute("data-icon", "filter-x");
  });
});

describe("every button has an svg.ic", () => {
  it("in the list screen", async () => {
    await show();
    for (const button of document.querySelectorAll("button")) {
      expect(button.querySelector("svg.ic"), button.outerHTML).not.toBeNull();
    }
  });
});
