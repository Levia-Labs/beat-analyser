// The Projects filter and sort state. It lives in the URL query (back button and links keep it),
// uses the API's own parameter names, and writes only what differs from the default.

export const SORT_KEYS = ["name", "key", "bpm", "duration", "created", "size"] as const;
export type SortKey = (typeof SORT_KEYS)[number];
export type Order = "asc" | "desc";
export type Status = "" | "built" | "pending";

export type ProjectsQuery = {
  q: string;
  key: string;
  status: Status;
  bpmMin: string;
  bpmMax: string;
  sort: SortKey;
  order: Order;
};

export const DEFAULT_QUERY: ProjectsQuery = {
  q: "",
  key: "",
  status: "",
  bpmMin: "",
  bpmMax: "",
  sort: "created",
  order: "desc",
};

export const PAGE_SIZE = 50;

const isSort = (value: string | null): value is SortKey =>
  SORT_KEYS.some((candidate) => candidate === value);

export function parseQuery(params: URLSearchParams): ProjectsQuery {
  const sort = params.get("sort");
  const status = params.get("status");
  return {
    q: params.get("q") ?? "",
    key: params.get("key") ?? "",
    status: status === "built" || status === "pending" ? status : "",
    bpmMin: params.get("bpm_min") ?? "",
    bpmMax: params.get("bpm_max") ?? "",
    sort: isSort(sort) ? sort : DEFAULT_QUERY.sort,
    order: params.get("order") === "asc" ? "asc" : DEFAULT_QUERY.order,
  };
}

function filters(query: ProjectsQuery): URLSearchParams {
  const params = new URLSearchParams();
  if (query.q) params.set("q", query.q);
  if (query.key) params.set("key", query.key);
  if (query.status) params.set("status", query.status);
  if (query.bpmMin) params.set("bpm_min", query.bpmMin);
  if (query.bpmMax) params.set("bpm_max", query.bpmMax);
  return params;
}

export function toSearchParams(query: ProjectsQuery): URLSearchParams {
  const params = filters(query);
  if (query.sort !== DEFAULT_QUERY.sort) params.set("sort", query.sort);
  if (query.order !== DEFAULT_QUERY.order) params.set("order", query.order);
  return params;
}

// The request always names sort and order, so the server default never decides what the header shows.
export function apiParams(query: ProjectsQuery, offset: number, limit: number): URLSearchParams {
  const params = filters(query);
  params.set("sort", query.sort);
  params.set("order", query.order);
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  return params;
}

// Text columns read best A to Z; numbers and dates, biggest and newest first.
export function toggleSort(query: ProjectsQuery, column: SortKey): ProjectsQuery {
  if (query.sort === column) return { ...query, order: query.order === "asc" ? "desc" : "asc" };
  return { ...query, sort: column, order: column === "name" || column === "key" ? "asc" : "desc" };
}

export function hasFilters(query: ProjectsQuery): boolean {
  return filters(query).size > 0;
}
