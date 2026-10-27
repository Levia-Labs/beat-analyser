import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App shell", () => {
  it("shows the logo and wordmark linking home", () => {
    render(
      <MemoryRouter initialEntries={["/nowhere"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: "Beat Analyser" })).toHaveAttribute("href", "/");
    expect(screen.getByText(/Nothing here/)).toBeInTheDocument();
  });
});
