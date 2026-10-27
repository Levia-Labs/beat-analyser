import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import ProjectList from "./ProjectList";

function show(props: Parameters<typeof ProjectList>[0]) {
  render(
    <MemoryRouter>
      <ProjectList {...props} />
    </MemoryRouter>,
  );
}

describe("ProjectList", () => {
  it("says so when there are no projects", () => {
    show({ projects: [], error: null, loading: false });
    expect(screen.getByText("No analyses yet.")).toBeInTheDocument();
  });

  it("links each project to its viewer and its ZIP", () => {
    show({
      projects: [
        { name: "song", done: true, meta: { key: "A minor", bpm: 120 } },
        { name: "half", done: false, meta: null },
      ],
      error: null,
      loading: false,
    });
    expect(screen.getByRole("link", { name: /song/ })).toHaveAttribute("href", "/p/song");
    expect(screen.getByRole("link", { name: "ZIP" })).toHaveAttribute("href", "/download/song");
    expect(screen.getByText("not built")).toBeInTheDocument();
    expect(screen.getByText(/A minor · 120 BPM/)).toBeInTheDocument();
  });

  it("shows a load error", () => {
    show({ projects: null, error: "Server unreachable", loading: false });
    expect(screen.getByRole("alert")).toHaveTextContent("Server unreachable");
  });
});
