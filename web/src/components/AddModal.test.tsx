import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getConfig, getProgress, postAnalysis } from "../api";
import AddModal from "./AddModal";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    getConfig: vi.fn(),
    postAnalysis: vi.fn(),
    getProgress: vi.fn(() => new Promise(() => {})),
  };
});
const post = vi.mocked(postAnalysis);

function Probe() {
  return <output data-testid="path">{useLocation().pathname}</output>;
}

async function open() {
  render(
    <MemoryRouter>
      <AddModal />
      <Probe />
    </MemoryRouter>,
  );
  await userEvent.click(screen.getByRole("button", { name: "Add" }));
  return screen.findByRole("dialog", { name: "Add project" });
}

async function startYoutube() {
  await userEvent.click(screen.getByLabelText("YouTube URL"));
  await userEvent.paste("https://youtu.be/dQw4w9WgXcQ");
  await userEvent.click(screen.getByRole("button", { name: "Analyse YouTube URL" }));
}

beforeEach(() => {
  post.mockReset();
  vi.mocked(getProgress).mockImplementation(() => new Promise(() => {}));
  vi.mocked(getConfig).mockResolvedValue({
    extensions: [".mp3"],
    max_upload_mb: 1,
  } as Awaited<ReturnType<typeof getConfig>>);
});

describe("Add modal", () => {
  it("opens titled Add project: the URL field first, an or divider, then the file zone", async () => {
    const dialog = await open();
    const url = screen.getByLabelText("YouTube URL");
    const file = screen.getByLabelText("Audio file");
    expect(url.compareDocumentPosition(file) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(dialog).toHaveTextContent("or");
    expect(screen.getByRole("button", { name: "Analyse YouTube URL" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Analyse file" })).toBeDisabled();
  });

  it("closes with Escape and with the close button while idle", async () => {
    await open();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Add" }));
    await userEvent.click(await screen.findByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("cannot be closed while a run is in flight", async () => {
    post.mockReturnValue(new Promise(() => {}));
    await open();
    await startYoutube();
    expect(await screen.findByText(/Analysing/)).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    fireEvent.pointerDown(document.body);
    expect(screen.getByRole("button", { name: "Close" })).toBeDisabled();
    expect(screen.getByRole("dialog", { name: "Add project" })).toBeInTheDocument();
  });

  it("on success it closes and goes to the new project's viewer", async () => {
    post.mockResolvedValue({ name: "my song" });
    await open();
    await startYoutube();
    await waitFor(() => expect(screen.getByTestId("path")).toHaveTextContent("/p/my%20song"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("every button in the modal carries an icon", async () => {
    const dialog = await open();
    for (const button of dialog.querySelectorAll("button")) {
      expect(button.querySelector("svg"), button.outerHTML).not.toBeNull();
    }
  });
});
