import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  deleteProject,
  getConfig,
  getEvents,
  login,
  logout,
  onUnauthorized,
  renameProject,
} from "./api";

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

  it("deletes with DELETE and accepts an empty 204", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteProject("a b")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith(
      "/projects/a%20b",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("renames with a JSON body and returns the new name", async () => {
    const fetchMock = vi.fn(
      async () => new Response(JSON.stringify({ name: "tune" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await expect(renameProject("song", "tune")).resolves.toEqual({ name: "tune" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/projects/song/rename",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ to: "tune" }) }),
    );
  });

  it("tells the unauthorized listener about a 401 and still rejects", async () => {
    answer(401, { error: "Sign in required" });
    const seen = vi.fn();
    const off = onUnauthorized(seen);
    await expect(getEvents("x")).rejects.toMatchObject({ status: 401 });
    expect(seen).toHaveBeenCalledTimes(1);
    off();
    await expect(getEvents("x")).rejects.toMatchObject({ status: 401 });
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it("does not treat other errors as a sign-in request", async () => {
    answer(404, { error: "Project not found" });
    const seen = vi.fn();
    const off = onUnauthorized(seen);
    await expect(getEvents("x")).rejects.toMatchObject({ status: 404 });
    off();
    expect(seen).not.toHaveBeenCalled();
  });

  it("posts the token to /login; a wrong token rejects without the listener", async () => {
    const fetchMock = vi.fn(
      async () => new Response(JSON.stringify({ error: "Wrong token" }), { status: 401 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const seen = vi.fn();
    const off = onUnauthorized(seen);
    await expect(login("nope")).rejects.toMatchObject({ message: "Wrong token", status: 401 });
    off();
    expect(seen).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "/login",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ token: "nope" }) }),
    );
  });

  it("logs out with POST /logout", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(logout()).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("/logout", expect.objectContaining({ method: "POST" }));
  });
});
