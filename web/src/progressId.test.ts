import { describe, expect, it } from "vitest";
import { progressId } from "./progressId";

describe("progressId", () => {
  it("is 32 lowercase hex characters and differs between calls", () => {
    const a = progressId();
    expect(a).toMatch(/^[0-9a-f]{32}$/);
    expect(progressId()).not.toBe(a);
  });
});
