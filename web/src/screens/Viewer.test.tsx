import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, deleteProject, getEvents, renameProject } from "../api";
import Viewer, { ViewerRoute } from "./Viewer";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getEvents: vi.fn(), deleteProject: vi.fn(), renameProject: vi.fn() };
});
const events = vi.mocked(getEvents);
const remove = vi.mocked(deleteProject);
const rename = vi.mocked(renameProject);
const project = {
  name: "song",
  meta: { key: "A minor", bpm: 120 },
  labels: {},
  tracks: { kick: [[0.5, 128]] as [number, number][] },
};

beforeEach(() => {
  events.mockReset();
  remove.mockReset();
  rename.mockReset();
});

function show(name: string) {
  render(
    <MemoryRouter>
      <Viewer name={name} />
    </MemoryRouter>,
  );
}

function at(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/" element={<p>home page</p>} />
        <Route path="/p/:name" element={<ViewerRoute />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Viewer", () => {
  it("shows the project's facts, ZIP link and lanes", async () => {
    events.mockResolvedValue({
      name: "song",
      meta: { key: "A minor", bpm: 120 },
      labels: {},
      tracks: { kick: [[0.5, 128]] },
    });
    show("song");
    expect(await screen.findByRole("heading", { name: "song" })).toBeInTheDocument();
    expect(screen.getByText(/A minor · 120 BPM/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "ZIP" })).toHaveAttribute("href", "/download/song");
    expect(screen.getByRole("button", { name: "kick, 1 events" })).toBeInTheDocument();
  });

  it("says when a project is not found or not analysed yet", async () => {
    events.mockRejectedValue(new ApiError("Project not found", 404));
    show("nope");
    expect(await screen.findByText("No analysed project called nope")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute("href", "/");
  });

  it("shows other errors as a banner", async () => {
    events.mockRejectedValue(new ApiError("Server unreachable", 0));
    show("song");
    expect(await screen.findByRole("alert")).toHaveTextContent("Server unreachable");
  });

  it("deletes after confirmation and goes home", async () => {
    events.mockResolvedValue(project);
    remove.mockResolvedValue(undefined);
    at("/p/song");
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("Delete song?");
    await userEvent.click(screen.getByRole("button", { name: "Delete song" }));
    expect(remove).toHaveBeenCalledWith("song");
    expect(await screen.findByText("home page")).toBeInTheDocument();
  });

  it("cancelling the dialog deletes nothing", async () => {
    events.mockResolvedValue(project);
    at("/p/song");
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(remove).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renames and opens the project under its new name", async () => {
    events.mockResolvedValue(project);
    rename.mockResolvedValue({ name: "tune" });
    at("/p/song");
    await userEvent.click(await screen.findByRole("button", { name: "Rename" }));
    const field = screen.getByLabelText("New name");
    await userEvent.clear(field);
    await userEvent.type(field, "tune");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(rename).toHaveBeenCalledWith("song", "tune");
    await vi.waitFor(() => expect(events).toHaveBeenLastCalledWith("tune"));
  });

  it("shows a refused rename", async () => {
    events.mockResolvedValue(project);
    rename.mockRejectedValue(new ApiError("A project called tune exists", 409));
    at("/p/song");
    await userEvent.click(await screen.findByRole("button", { name: "Rename" }));
    await userEvent.clear(screen.getByLabelText("New name"));
    await userEvent.type(screen.getByLabelText("New name"), "tune");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("A project called tune exists");
  });
});
