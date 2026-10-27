// Every URL the SPA calls lives here. Errors become ApiError with the server's own `error`
// text, so the UI shows what Flask said.

export type Meta = { name?: string; key?: string; bpm?: number } | null;
export type Project = { name: string; done: boolean; meta: Meta };
export type Config = { extensions: string[]; max_upload_mb: number };
export type Row = [number, number];
export type Events = {
  name: string;
  meta: Meta;
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

async function get<T>(url: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError("Server unreachable", 0);
  }
  const body = parseJson(await response.text());
  if (!response.ok) throw new ApiError(errorText(body, response.status), response.status);
  return body as T;
}

export const getConfig = () => get<Config>("/config");
export const getProjects = () => get<Project[]>("/projects");
export const getEvents = (name: string) =>
  get<Events>(`/projects/${encodeURIComponent(name)}/events`);

// XMLHttpRequest, not fetch: fetch reports no upload progress.
export function postAnalysis(
  endpoint: "/upload" | "/download_yt",
  body: FormData | string,
  onProgress: (fraction: number) => void,
): Promise<{ name: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", endpoint);
    xhr.setRequestHeader("Accept", "application/json");
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
        reject(new ApiError(errorText(parsed, xhr.status), xhr.status));
      }
    };
    xhr.send(body);
  });
}
