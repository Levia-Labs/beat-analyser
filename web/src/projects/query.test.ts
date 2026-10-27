import { describe, expect, it } from "vitest";
import { apiParams, DEFAULT_QUERY, parseQuery, toggleSort, toSearchParams } from "./query";

describe("project query <-> URL", () => {
  it("an empty URL is the default query: Created, newest first", () => {
    expect(parseQuery(new URLSearchParams(""))).toEqual(DEFAULT_QUERY);
    expect(DEFAULT_QUERY.sort).toBe("created");
    expect(DEFAULT_QUERY.order).toBe("desc");
  });

  it("reads every filter from the URL", () => {
    const query = parseQuery(
      new URLSearchParams(
        "q=song&key=A+minor&status=built&bpm_min=90&bpm_max=140&sort=bpm&order=asc",
      ),
    );
    expect(query).toEqual({
      q: "song",
      key: "A minor",
      status: "built",
      bpmMin: "90",
      bpmMax: "140",
      sort: "bpm",
      order: "asc",
    });
  });

  it("falls back to the defaults on values the API would refuse", () => {
    const query = parseQuery(new URLSearchParams("sort=weight&order=sideways&status=maybe"));
    expect(query.sort).toBe("created");
    expect(query.order).toBe("desc");
    expect(query.status).toBe("");
  });

  it("writes only what differs from the default, and round-trips", () => {
    expect(toSearchParams(DEFAULT_QUERY).toString()).toBe("");
    const query = { ...DEFAULT_QUERY, q: "a b", key: "minor", bpmMin: "80", sort: "name" as const };
    const params = toSearchParams(query);
    expect(params.get("q")).toBe("a b");
    expect(params.get("bpm_min")).toBe("80");
    expect(params.has("bpm_max")).toBe(false);
    expect(params.get("sort")).toBe("name");
    expect(params.has("order")).toBe(false);
    expect(parseQuery(params)).toEqual(query);
  });

  it("builds the API request with sort, order, offset and limit always set", () => {
    const params = apiParams({ ...DEFAULT_QUERY, q: "x", bpmMax: "150" }, 50, 50);
    expect(params.get("q")).toBe("x");
    expect(params.get("bpm_max")).toBe("150");
    expect(params.has("key")).toBe(false);
    expect(params.get("sort")).toBe("created");
    expect(params.get("order")).toBe("desc");
    expect(params.get("offset")).toBe("50");
    expect(params.get("limit")).toBe("50");
  });

  it("toggles sort: same column flips, a new text column starts ascending, a number descending", () => {
    expect(toggleSort(DEFAULT_QUERY, "created").order).toBe("asc");
    expect(toggleSort({ ...DEFAULT_QUERY, order: "asc" }, "created").order).toBe("desc");
    expect(toggleSort(DEFAULT_QUERY, "name")).toMatchObject({ sort: "name", order: "asc" });
    expect(toggleSort(DEFAULT_QUERY, "bpm")).toMatchObject({ sort: "bpm", order: "desc" });
  });
});
