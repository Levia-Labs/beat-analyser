import { render, screen } from "@testing-library/react";
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
    expect(brand.closest("header")).toHaveClass("masthead");
    expect(brand).toHaveClass("brand");
  });

  it("renders no tabs and no phone bottom bar, on any route", () => {
    for (const path of ["/", "/p/x", "/nowhere"]) {
      const { unmount } = at(path);
      expect(screen.queryByRole("navigation")).toBeNull();
      expect(screen.queryByRole("link", { name: "Add" })).toBeNull();
      unmount();
    }
  });

  it("the list route has the Add button that opens the modal", () => {
    at("/");
    expect(screen.getByRole("heading", { name: "Projects" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add" })).toBeInTheDocument();
  });

  it("has no /add route any more", () => {
    at("/add");
    expect(screen.getByRole("heading", { name: "Not found" })).toBeInTheDocument();
  });

  it("shows Not found on an unknown route", () => {
    at("/nowhere");
    expect(screen.getByRole("heading", { name: "Not found" })).toBeInTheDocument();
  });
});
