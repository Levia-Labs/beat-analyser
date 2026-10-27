import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getConfig, getProjects } from "../api";
import ProjectsScreen from "./ProjectsScreen";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getProjects: vi.fn(), getConfig: vi.fn() };
});

async function show(path = "/") {
  render(
    <MemoryRouter initialEntries={[path]}>
      <ProjectsScreen />
    </MemoryRouter>,
  );
  await screen.findByRole("link", { name: "a" });
}

// hidden: an open modal popover marks everything outside it aria-hidden.
const trigger = (name: string) => screen.getByRole("button", { name, hidden: true });

async function openKeys() {
  await userEvent.click(trigger("Filter Key"));
  return screen.findByRole("dialog", { name: "Key filter" });
}

const labels = (panel: HTMLElement) =>
  within(panel)
    .getAllByRole("checkbox")
    .map((box) => box.parentElement?.querySelector("label")?.textContent);

beforeEach(() => {
  vi.mocked(getProjects).mockReset();
  vi.mocked(getConfig).mockReset();
  vi.mocked(getProjects).mockResolvedValue({
    items: [
      { name: "a", done: true, key: "A minor", bpm: 120, duration: 185, created: null, size: 5 },
    ],
    total: 7,
  });
});

describe("key search", () => {
  it("is a search input above the checklist, empty at first, listing every key", async () => {
    await show();
    const panel = await openKeys();
    expect(within(panel).getByRole("searchbox", { name: "Search keys" })).toHaveValue("");
    expect(within(panel).getAllByRole("checkbox")).toHaveLength(26);
  });

  it("matches the visible label, ignoring case and spaces", async () => {
    await show();
    const panel = await openKeys();
    const box = within(panel).getByRole("searchbox", { name: "Search keys" });
    await userEvent.type(box, "A M");
    expect(labels(panel)).toEqual(["A major", "A minor"]);
    await userEvent.clear(box);
    await userEvent.type(box, "maj");
    const shown = labels(panel);
    expect(shown).toHaveLength(13);
    expect(shown[0]).toBe("Major");
    expect(shown.slice(1).every((label) => label?.endsWith(" major"))).toBe(true);
  });

  it("keeps checked keys listed while the search hides them", async () => {
    await show();
    const panel = await openKeys();
    await userEvent.click(within(panel).getByRole("checkbox", { name: "C major" }));
    await userEvent.type(within(panel).getByRole("searchbox", { name: "Search keys" }), "zzz");
    expect(labels(panel)).toEqual(["C major"]);
  });

  it("is local state: not in the URL, reset when the popover closes", async () => {
    await show();
    const panel = await openKeys();
    await userEvent.type(within(panel).getByRole("searchbox", { name: "Search keys" }), "minor");
    await userEvent.keyboard("{Escape}");
    const again = await openKeys();
    expect(within(again).getByRole("searchbox", { name: "Search keys" })).toHaveValue("");
    expect(within(again).getAllByRole("checkbox")).toHaveLength(26);
  });

  it("the Status checklist has no search", async () => {
    await show();
    await userEvent.click(trigger("Filter Status"));
    const panel = await screen.findByRole("dialog", { name: "Status filter" });
    expect(within(panel).queryByRole("searchbox")).toBeNull();
  });
});
