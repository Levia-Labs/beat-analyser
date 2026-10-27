import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ErrorBoundary from "./ErrorBoundary";

function Boom(): never {
  throw new Error("kaboom");
}

function Where() {
  return <p data-testid="where">{useLocation().pathname}</p>;
}

function shell() {
  render(
    <MemoryRouter initialEntries={["/bad"]}>
      <Where />
      <Link to="/fine">go fine</Link>
      <ErrorBoundary>
        <Routes>
          <Route path="/bad" element={<Boom />} />
          <Route path="/fine" element={<p>fine page</p>} />
          <Route path="/" element={<p>home page</p>} />
        </Routes>
      </ErrorBoundary>
    </MemoryRouter>,
  );
}

// React re-dispatches caught render errors on window; vitest would report them as unhandled.
function swallow(event: ErrorEvent) {
  event.preventDefault();
}

beforeEach(() => {
  // React logs caught render errors; keep the test output readable.
  vi.spyOn(console, "error").mockImplementation(() => {});
  window.addEventListener("error", swallow);
});
afterEach(() => {
  window.removeEventListener("error", swallow);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ErrorBoundary", () => {
  it("renders its page with the message and no stack when a child throws", () => {
    shell();
    expect(screen.getByRole("heading", { name: "Something went wrong" })).toBeInTheDocument();
    expect(screen.getByText("kaboom")).toBeInTheDocument();
    expect(screen.queryByText(/at Boom/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload" }).querySelector("svg")).not.toBeNull();
    expect(screen.getByRole("button", { name: "Projects" }).querySelector("svg")).not.toBeNull();
  });

  it("Reload calls location.reload", async () => {
    const reload = vi.fn();
    vi.stubGlobal("location", { ...window.location, reload });
    shell();
    await userEvent.click(screen.getByRole("button", { name: "Reload" }));
    expect(reload).toHaveBeenCalledOnce();
  });

  it("Projects clears the error and navigates home", async () => {
    shell();
    await userEvent.click(screen.getByRole("button", { name: "Projects" }));
    expect(screen.getByTestId("where")).toHaveTextContent("/");
    expect(screen.getByText("home page")).toBeInTheDocument();
    expect(screen.queryByText("kaboom")).not.toBeInTheDocument();
  });

  it("clears the error on a route change", async () => {
    shell();
    await userEvent.click(screen.getByRole("link", { name: "go fine" }));
    expect(screen.getByText("fine page")).toBeInTheDocument();
    expect(screen.queryByText("kaboom")).not.toBeInTheDocument();
  });
});
