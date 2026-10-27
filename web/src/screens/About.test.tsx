import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getConfig: vi.fn(() => new Promise(() => {})) };
});

beforeEach(() => {
  document.title = "Beat Analyser";
});

function at(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("About page", () => {
  it("titles the tab", () => {
    at("/about");
    expect(document.title).toBe("About · Beat Analyser");
  });

  it("shows the sections in order under one h1", () => {
    at("/about");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("About");
    const titles = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(titles).toEqual([
      "What it is",
      "How it works",
      "What you get",
      "Runs on your machine",
      "For agents",
      "Links",
    ]);
  });

  it("explains in three ordered steps", () => {
    at("/about");
    const steps = screen.getByRole("list", { name: "Steps" });
    expect(steps.tagName).toBe("OL");
    expect(steps.querySelectorAll("li")).toHaveLength(3);
  });

  it("names the MCP endpoint", () => {
    at("/about");
    expect(screen.getByText(/http:\/\/localhost:8001\/mcp/)).toBeInTheDocument();
  });

  it("opens both links in a new tab without leaking the opener", () => {
    at("/about");
    for (const [name, href] of [
      ["GitHub", "https://github.com/Levia-Labs/beat-analyser"],
      ["Levia Labs", "https://levialabs.net"],
    ]) {
      const link = screen.getByRole("link", { name });
      expect(link).toHaveAttribute("href", href);
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
      expect(link.querySelector("svg.ic[data-icon='external-link']")).not.toBeNull();
    }
  });

  it("shows the build version, dev by default", () => {
    at("/about");
    expect(screen.getByText("Version dev")).toBeInTheDocument();
  });

  it("links back to the projects", () => {
    at("/about");
    expect(screen.getByRole("link", { name: "Projects" })).toHaveAttribute("href", "/");
  });
});

describe("About link in the masthead", () => {
  it("is an icon link named About that goes to /about", () => {
    at("/");
    const link = screen.getByRole("link", { name: "About" });
    expect(link).toHaveAttribute("href", "/about");
    expect(link.closest("header")).toHaveClass("masthead");
    expect(link.querySelector("svg.ic[data-icon='info']")).not.toBeNull();
  });
});
