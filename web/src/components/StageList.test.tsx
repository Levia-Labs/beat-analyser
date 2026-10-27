import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Progress } from "../api";
import StageList from "./StageList";

const base: Progress = {
  stages: ["download", "decode", "rhythm", "notes", "key", "package"],
  stage: "download",
  done: 0,
  total: 6,
  fraction: 0.42,
  finished: false,
  error: null,
  elapsed: 3,
};

describe("StageList", () => {
  it("marks the current stage and shows the download percentage", () => {
    render(<StageList progress={base} />);
    expect(screen.getByText(/Download/).closest("li")).toHaveAttribute("aria-current", "step");
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("ticks finished stages", () => {
    render(<StageList progress={{ ...base, stage: "notes", done: 3, fraction: null }} />);
    expect(screen.getByText("Rhythm").closest("li")).toHaveClass("done");
    expect(screen.getByText("Notes").closest("li")).toHaveAttribute("aria-current", "step");
    expect(screen.getByText("Key").closest("li")).toHaveClass("todo");
  });
});
