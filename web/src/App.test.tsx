import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    getConfig: vi.fn(() => new Promise(() => {})),
    getProjects: vi.fn(() => new Promise(() => {})),
    getEvents: vi.fn(() => new Promise(() => {})),
  };
});

function at(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App shell", () => {
  it("centers the logo and wordmark and links home", () => {
    at("/");
    const brand = screen.getByRole("link", { name: "Beat Analyser" });
    expect(brand).toHaveAttribute("href", "/");
    expect(brand.closest("header")).toHaveClass("top");
    expect(brand).toHaveClass("brand");
  });

  it("has Projects and Add tabs that route", () => {
    at("/");
    const nav = screen.getByRole("navigation", { name: "Sections" });
    expect(within(nav).getByRole("link", { name: "Projects" })).toHaveAttribute("href", "/");
    expect(within(nav).getByRole("link", { name: "Add" })).toHaveAttribute("href", "/add");
  });

  it("marks Projects active on / and on a viewer route, Add on /add", () => {
    const { unmount } = at("/p/x");
    let nav = screen.getByRole("navigation", { name: "Sections" });
    expect(within(nav).getByRole("link", { name: "Projects" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(within(nav).getByRole("link", { name: "Add" })).not.toHaveAttribute("aria-current");
    unmount();
    at("/add");
    nav = screen.getByRole("navigation", { name: "Sections" });
    expect(within(nav).getByRole("link", { name: "Add" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getByRole("link", { name: "Projects" })).not.toHaveAttribute("aria-current");
  });

  it("renders the same links in the phone bottom bar", () => {
    at("/add");
    const bar = screen.getByRole("navigation", { name: "Sections, bottom bar" });
    expect(bar).toHaveClass("bottom-bar");
    expect(within(bar).getByRole("link", { name: "Projects" })).toHaveAttribute("href", "/");
    expect(within(bar).getByRole("link", { name: "Add" })).toHaveAttribute("href", "/add");
    expect(within(bar).getByRole("link", { name: "Add" })).toHaveAttribute("aria-current", "page");
  });

  it("renders the analyse card on /add", () => {
    at("/add");
    expect(screen.getByRole("button", { name: "Analyse" })).toBeInTheDocument();
  });

  it("shows Not found on an unknown route, with the tabs still there", () => {
    at("/nowhere");
    expect(screen.getByRole("heading", { name: "Not found" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Sections" })).toBeInTheDocument();
  });
});
