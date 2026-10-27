import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, deleteProject, getProjects, type ProjectItem } from "../api";
import ProjectsScreen from "./ProjectsScreen";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getProjects: vi.fn(), deleteProject: vi.fn(), renameProject: vi.fn() };
});

const list = vi.mocked(getProjects);
const remove = vi.mocked(deleteProject);

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

beforeEach(() => {
  list.mockReset();
  remove.mockReset();
});

describe("Bulk delete under a refusing server", () => {
  it.each([401, 429])("stops at the first %i and reports the rest", async (status) => {
    list.mockResolvedValue({ items: [item("a"), item("b"), item("c")], total: 3 });
    remove.mockImplementation((name) =>
      name === "b" ? Promise.reject(new ApiError("Refused", status)) : Promise.resolve(),
    );
    render(
      <MemoryRouter>
        <ProjectsScreen />
      </MemoryRouter>,
    );
    await screen.findByRole("link", { name: "a" });
    await userEvent.click(screen.getByRole("checkbox", { name: "Select all loaded" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete selected (3)" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Delete" }),
    );
    const banner = await screen.findByRole("alert");
    expect(remove.mock.calls).toEqual([["a"], ["b"]]);
    expect(banner).toHaveTextContent("b: Refused");
    expect(banner).toHaveTextContent("not attempted: c");
  });
});
