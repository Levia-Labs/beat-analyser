import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api";
import SignIn from "./SignIn";

describe("SignIn", () => {
  it("has a password field and a Sign in button", () => {
    render(<SignIn submit={vi.fn()} onDone={vi.fn()} />);
    expect(screen.getByLabelText("Access token")).toHaveAttribute("type", "password");
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDisabled();
  });

  it("submits the token and reports success", async () => {
    const submit = vi.fn(async () => {});
    const onDone = vi.fn();
    render(<SignIn submit={submit} onDone={onDone} />);
    await userEvent.type(screen.getByLabelText("Access token"), "tok");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(submit).toHaveBeenCalledWith("tok");
  });

  it("shows the server's refusal and stays", async () => {
    const submit = vi.fn(async () => {
      throw new ApiError("Wrong token", 401);
    });
    const onDone = vi.fn();
    render(<SignIn submit={submit} onDone={onDone} />);
    await userEvent.type(screen.getByLabelText("Access token"), "bad");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Wrong token");
    expect(onDone).not.toHaveBeenCalled();
  });
});
