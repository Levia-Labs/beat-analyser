import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => {
  vi.unstubAllGlobals();
});

// A tiny server: /projects answers 401 until /login has been called.
function server(auth: boolean) {
  let signedIn = false;
  const calls: string[] = [];
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push(`${init?.method ?? "GET"} ${url}`);
    if (url === "/config") return json({ extensions: [".wav"], max_upload_mb: 200, auth });
    if (url === "/login") {
      const ok = JSON.parse(String(init?.body)).token === "tok";
      signedIn = ok;
      return ok ? new Response(null, { status: 204 }) : json({ error: "Wrong token" }, 401);
    }
    if (url === "/logout") {
      signedIn = false;
      return new Response(null, { status: 204 });
    }
    if (url.startsWith("/projects")) {
      return signedIn || !auth
        ? json({ items: [], total: 0 })
        : json({ error: "Sign in required" }, 401);
    }
    return json({ error: "nope" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

function open() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App sign-in", () => {
  it("shows the Sign in page on a 401, signs in and retries", async () => {
    const calls = server(true);
    open();
    const field = await screen.findByLabelText("Access token");
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    const before = calls.filter((c) => c.startsWith("GET /projects")).length;
    await userEvent.type(field, "tok");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(screen.queryByLabelText("Access token")).not.toBeInTheDocument());
    expect(calls.filter((c) => c.startsWith("GET /projects")).length).toBeGreaterThan(before);
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
  });

  it("keeps the page and shows the refusal on a wrong token", async () => {
    server(true);
    open();
    await userEvent.type(await screen.findByLabelText("Access token"), "bad");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Wrong token")).toBeInTheDocument();
    expect(screen.getByLabelText("Access token")).toBeInTheDocument();
  });

  it("Sign out posts /logout and returns to the Sign in page", async () => {
    const calls = server(true);
    open();
    await userEvent.type(await screen.findByLabelText("Access token"), "tok");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await userEvent.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(await screen.findByLabelText("Access token")).toBeInTheDocument();
    expect(calls).toContain("POST /logout");
  });

  it("shows no Sign out and no Sign in page when the server has no token", async () => {
    const calls = server(false);
    open();
    await waitFor(() => expect(calls).toContain("GET /config"));
    await waitFor(() => expect(calls.some((c) => c.startsWith("GET /projects"))).toBe(true));
    expect(screen.queryByRole("button", { name: "Sign out" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Access token")).not.toBeInTheDocument();
  });
});
