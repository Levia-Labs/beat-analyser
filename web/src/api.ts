// Every URL the SPA calls lives here. Errors become ApiError with the server's own `error`
// text, so the UI shows what Flask said.

export type Meta = { name?: string; key?: string; bpm?: number } | null;
export type ProjectItem = {
  name: string;
  done: boolean;
  key: string | null;
  bpm: number | null;
  duration: number | null;
  created: string | null;
  size: number;
};
export type ProjectsPage = { items: ProjectItem[]; total: number };
export type Config = { extensions: string[]; max_upload_mb: number; auth?: boolean };
export type Progress = {
  stages: string[];
  stage: string;
  done: number;
  total: number;
  fraction: number | null;
  finished: boolean;
  error: string | null;
  elapsed: number;
};
export type Row = [number, number];
export type Events = {
  name: string;
  meta: Meta;
  // Optional: an older server's response lacks them.
  duration?: number | null;
  created?: string | null;
  size?: number;
  labels: Record<string, string>;
  tracks: Record<string, Row[]>;
};

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// With WEB_TOKEN set the server answers 401 to any API call without the session cookie; the
// app shows the Sign in page instead of each screen's own error.
const unauthorizedListeners = new Set<() => void>();
export function onUnauthorized(listener: () => void): () => void {
  unauthorizedListeners.add(listener);
  return () => {
    unauthorizedListeners.delete(listener);
  };
}
// A wrong token at /login is the form's own error, not a request to sign in.
function notifyUnauthorized(url: string) {
  if (url !== "/login") unauthorizedListeners.forEach((listener) => listener());
}

function errorText(body: unknown, status: number): string {
  if (body && typeof body === "object" && "error" in body) return String(body.error);
  return `Server answered ${status}`;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (typeof init.body === "string") headers["Content-Type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(url, { ...init, headers });
  } catch {
    throw new ApiError("Server unreachable", 0);
  }
  if (response.status === 204) return undefined as T;
  let text: string;
  try {
    text = await response.text();
  } catch {
    throw new ApiError("Server unreachable", 0);
  }
  const body = parseJson(text);
  if (response.status === 401) notifyUnauthorized(url);
  if (!response.ok) throw new ApiError(errorText(body, response.status), response.status);
  if (body === null) {
    throw new ApiError(`Server answered ${response.status} without JSON`, response.status);
  }
  return body as T;
}

const get = <T>(url: string) => request<T>(url);

export const deleteProject = (name: string) =>
  request<void>(`/projects/${encodeURIComponent(name)}`, { method: "DELETE" });
export const renameProject = (name: string, to: string) =>
  request<{ name: string }>(`/projects/${encodeURIComponent(name)}/rename`, {
    method: "POST",
    body: JSON.stringify({ to }),
  });
export const login = (token: string) =>
  request<void>("/login", { method: "POST", body: JSON.stringify({ token }) });
export const logout = () => request<void>("/logout", { method: "POST" });
export const getConfig = () => get<Config>("/config");
export const getProgress = (id: string) => get<Progress>(`/progress/${id}`);
export const getProjects = (params: URLSearchParams) =>
  get<ProjectsPage>(`/projects?${params.toString()}`);
export const getEvents = (name: string) =>
  get<Events>(`/projects/${encodeURIComponent(name)}/events`);

// XMLHttpRequest, not fetch: fetch reports no upload progress.
export function postAnalysis(
  endpoint: "/upload" | "/download_yt",
  body: FormData | string,
  onProgress: (fraction: number) => void,
  progressId?: string,
): Promise<{ name: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", endpoint);
    xhr.setRequestHeader("Accept", "application/json");
    if (progressId) xhr.setRequestHeader("X-Progress-Id", progressId);
    if (typeof body === "string") xhr.setRequestHeader("Content-Type", "application/json");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.upload.onload = () => onProgress(1);
    xhr.onerror = () => reject(new ApiError("Server unreachable", 0));
    xhr.onload = () => {
      const parsed = parseJson(xhr.responseText);
      if (xhr.status === 200 && parsed && typeof parsed === "object" && "name" in parsed) {
        resolve({ name: String(parsed.name) });
      } else {
        if (xhr.status === 401) notifyUnauthorized(endpoint);
        reject(new ApiError(errorText(parsed, xhr.status), xhr.status));
      }
    };
    xhr.send(body);
  });
}
