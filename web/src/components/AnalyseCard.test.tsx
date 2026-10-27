import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, getProgress, postAnalysis, type Progress } from "../api";
import AnalyseCard from "./AnalyseCard";

// ES module exports cannot be spied on in place; mock the module, keep the real ApiError.
vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, postAnalysis: vi.fn(), getProgress: vi.fn(() => new Promise(() => {})) };
});
const post = vi.mocked(postAnalysis);
const poll = vi.mocked(getProgress);

const config = { extensions: [".mp3", ".wav"], max_upload_mb: 1 };

beforeEach(() => {
  post.mockReset();
  poll.mockReset();
  poll.mockImplementation(() => new Promise(() => {}));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// Pasting is one event; typing 28 keys one by one took long enough under load to hit timeouts.
async function paste(text: string) {
  await userEvent.click(screen.getByLabelText("YouTube URL"));
  await userEvent.paste(text);
}

function audio(bytes: number) {
  return new File([new Uint8Array(bytes)], "song.mp3", { type: "audio/mpeg" });
}

describe("AnalyseCard", () => {
  it("enables Analyse only once a file is chosen", async () => {
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    const go = screen.getByRole("button", { name: "Analyse" });
    expect(go).toBeDisabled();
    await userEvent.upload(screen.getByLabelText("Audio file"), audio(10));
    expect(go).toBeEnabled();
    expect(screen.getByText("song.mp3")).toBeInTheDocument();
  });

  it("refuses a file over the limit without sending it", async () => {
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    await userEvent.upload(screen.getByLabelText("Audio file"), audio(2 * 1024 * 1024));
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    expect(screen.getByRole("alert")).toHaveTextContent("limit is 1 MB");
    expect(post).not.toHaveBeenCalled();
  });

  it("reports the new project's name when the upload succeeds", async () => {
    post.mockResolvedValue({ name: "song" });
    const onDone = vi.fn();
    render(<AnalyseCard config={config} onDone={onDone} />);
    await userEvent.upload(screen.getByLabelText("Audio file"), audio(10));
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    await waitFor(() => expect(onDone).toHaveBeenCalledWith("song"));
  });

  it("shows the server's error and keeps the form for a retry", async () => {
    post.mockRejectedValue(new ApiError("Processing failed: boom", 500));
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    await userEvent.upload(screen.getByLabelText("Audio file"), audio(10));
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Processing failed: boom");
    expect(screen.getByRole("button", { name: "Analyse" })).toBeEnabled();
  });

  it("ignores a drop while an upload is running", async () => {
    post.mockReturnValue(new Promise(() => {}));
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    await userEvent.upload(screen.getByLabelText("Audio file"), audio(10));
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    const other = new File([new Uint8Array(5)], "other.mp3", { type: "audio/mpeg" });
    fireEvent.drop(screen.getByText("song.mp3").closest("label")!, {
      dataTransfer: { files: [other] },
    });
    expect(screen.getByRole("button", { name: "Analyse" })).toBeDisabled();
    expect(screen.getByText("song.mp3")).toBeInTheDocument();
    expect(screen.queryByText("other.mp3")).not.toBeInTheDocument();
  });

  it("keeps a YouTube request in the analysing phase whatever the progress", async () => {
    post.mockReturnValue(new Promise(() => {}));
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    await userEvent.click(screen.getByRole("tab", { name: "YouTube" }));
    await paste("https://youtu.be/dQw4w9WgXcQ");
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    const report = post.mock.calls[0]![2];
    act(() => report(0.5));
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.getByText(/Analysing/)).toBeInTheDocument();
  });

  it("sends a YouTube URL as JSON to /download_yt", async () => {
    post.mockResolvedValue({ name: "dQw4w9WgXcQ" });
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    await userEvent.click(screen.getByRole("tab", { name: "YouTube" }));
    await paste("https://youtu.be/dQw4w9WgXcQ");
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    // The resolved POST sets state after the click returns; wait for the card to settle.
    await waitFor(() => expect(screen.getByRole("button", { name: "Analyse" })).toBeEnabled());
    expect(post).toHaveBeenCalledWith(
      "/download_yt",
      JSON.stringify({ url: "https://youtu.be/dQw4w9WgXcQ" }),
      expect.any(Function),
      expect.stringMatching(/^[0-9a-f]{32}$/),
    );
  });

  it("still submits where crypto.randomUUID is missing (plain http)", async () => {
    vi.stubGlobal("crypto", { getRandomValues: crypto.getRandomValues.bind(crypto) });
    post.mockResolvedValue({ name: "x" });
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    await userEvent.click(screen.getByRole("tab", { name: "YouTube" }));
    await paste("https://youtu.be/dQw4w9WgXcQ");
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    await waitFor(() => expect(post).toHaveBeenCalled());
    expect(post.mock.calls[0]![3]).toMatch(/^[0-9a-f]{32}$/);
    await waitFor(() => expect(screen.getByRole("button", { name: "Analyse" })).toBeEnabled());
  });

  it("polls the server once a second while the POST is open, then stops", async () => {
    vi.useFakeTimers();
    let finish: (value: { name: string }) => void = () => {};
    post.mockReturnValue(new Promise((resolve) => (finish = resolve)));
    const running: Progress = {
      stages: ["download", "decode", "rhythm"],
      stage: "decode",
      done: 1,
      total: 3,
      fraction: null,
      finished: false,
      error: null,
      elapsed: 1,
    };
    poll.mockResolvedValue(running);
    render(<AnalyseCard config={config} onDone={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "YouTube" }));
    fireEvent.change(screen.getByLabelText("YouTube URL"), {
      target: { value: "https://youtu.be/dQw4w9WgXcQ" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyse" }));
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(poll).toHaveBeenCalledTimes(1);
    expect(poll).toHaveBeenCalledWith(post.mock.calls[0]![3]);
    expect(screen.getByText("Decode").closest("li")).toHaveAttribute("aria-current", "step");
    await act(async () => finish({ name: "x" }));
    const calls = poll.mock.calls.length;
    await act(() => vi.advanceTimersByTimeAsync(3000));
    expect(poll).toHaveBeenCalledTimes(calls);
  });
});
