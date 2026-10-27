import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getConfig, getEvents } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

function answer(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  );
}

describe("api", () => {
  it("returns the parsed body", async () => {
    answer(200, { extensions: [".wav"], max_upload_mb: 200 });
    await expect(getConfig()).resolves.toEqual({ extensions: [".wav"], max_upload_mb: 200 });
  });

  it("raises the server's error text with its status", async () => {
    answer(404, { error: "Project not found" });
    await expect(getEvents("x")).rejects.toEqual(new ApiError("Project not found", 404));
  });

  it("quotes the status when the body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>", { status: 502 })),
    );
    await expect(getConfig()).rejects.toMatchObject({
      message: "Server answered 502",
      status: 502,
    });
  });

  it("says unreachable when fetch fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("down"))),
    );
    await expect(getConfig()).rejects.toMatchObject({ message: "Server unreachable", status: 0 });
  });

  it("says unreachable when the connection drops mid-body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        text: () => Promise.reject(new TypeError("cut")),
      })),
    );
    await expect(getConfig()).rejects.toMatchObject({ message: "Server unreachable", status: 0 });
  });

  it("refuses a 200 that is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>", { status: 200 })),
    );
    await expect(getConfig()).rejects.toMatchObject({
      message: "Server answered 200 without JSON",
      status: 200,
    });
  });

  it("encodes the project name", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await getEvents("a b");
    expect(fetchMock).toHaveBeenCalledWith("/projects/a%20b/events", expect.anything());
  });
});
