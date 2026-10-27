import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { getEvents } from "../api";
import Viewer from "../screens/Viewer";
import AnalyseCard from "./AnalyseCard";
import ConfirmDialog from "./ConfirmDialog";
import RenameForm from "./RenameForm";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, getEvents: vi.fn() };
});

// Lane buttons are text toggles inside the timeline canvas; every other button gets an icon.
function plainButtons(root: HTMLElement) {
  return Array.from(root.querySelectorAll("button")).filter((b) => !b.classList.contains("lane"));
}

function expectIcons(root: HTMLElement) {
  const buttons = plainButtons(root);
  expect(buttons.length).toBeGreaterThan(0);
  for (const button of buttons) {
    expect(button.querySelector("svg"), button.outerHTML).not.toBeNull();
    expect(button.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
    if (!button.textContent?.trim()) expect(button).toHaveAttribute("aria-label");
  }
}

describe("icons on buttons", () => {
  it("the Add card", () => {
    const { container } = render(
      <AnalyseCard config={{ extensions: [".mp3"], max_upload_mb: 1 }} onDone={vi.fn()} />,
    );
    expectIcons(container);
  });

  it("the viewer header, zoom controls, rename form and delete dialog", async () => {
    vi.mocked(getEvents).mockResolvedValue({
      name: "song",
      meta: { key: "A minor", bpm: 120 },
      labels: {},
      tracks: { kick: [[0.5, 128]] },
    });
    const { container } = render(
      <MemoryRouter>
        <Viewer name="song" />
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "song" });
    expectIcons(container);
    expect(screen.getByRole("link", { name: "ZIP" }).querySelector("svg")).not.toBeNull();
    expect(screen.getByRole("link", { name: "Projects" }).querySelector("svg")).not.toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Rename" }));
    expectIcons(container);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expectIcons(container);
  });

  it("the dialogs and forms alone", () => {
    const { container } = render(
      <>
        <ConfirmDialog
          title="t"
          body="b"
          confirm="Delete x"
          busy={false}
          onConfirm={vi.fn()}
          onCancel={vi.fn()}
        />
        <RenameForm name="a" busy={false} onSave={vi.fn()} onCancel={vi.fn()} />
      </>,
    );
    expectIcons(container);
  });
});
