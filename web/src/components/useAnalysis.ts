import { useEffect, useState } from "react";
import { ApiError, getProgress, postAnalysis, type Config, type Progress } from "../api";
import { progressId as newProgressId } from "../progressId";

const MB = 1024 * 1024;

export type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; fraction: number }
  | { kind: "analysing"; since: number }
  | { kind: "error"; message: string };

export function sizeLabel(bytes: number): string {
  return bytes >= MB
    ? `${(bytes / MB).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

// One run at a time: a file upload or a YouTube URL. The hook owns the phase, the progress id and
// the server's stage list; the two input parts only call runFile or runUrl.
export function useAnalysis(config: Config | null, onDone: (name: string) => void) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [progressId, setProgressId] = useState<string | null>(null);
  const [server, setServer] = useState<Progress | null>(null);
  const busy = phase.kind === "uploading" || phase.kind === "analysing";

  // Poll while the POST is open. A failed poll is ignored: the POST's own answer decides.
  useEffect(() => {
    if (!progressId) return;
    let live = true;
    const timer = window.setInterval(() => {
      getProgress(progressId).then(
        (state) => live && setServer(state),
        () => undefined,
      );
    }, 1000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [progressId]);

  async function run(endpoint: "/upload" | "/download_yt", body: FormData | string) {
    const id = newProgressId();
    setPhase(
      endpoint === "/upload"
        ? { kind: "uploading", fraction: 0 }
        : { kind: "analysing", since: Date.now() },
    );
    setProgressId(id);
    setServer(null);
    try {
      const { name } = await postAnalysis(
        endpoint,
        body,
        (fraction) =>
          setPhase((current) =>
            fraction < 1 && endpoint === "/upload"
              ? { kind: "uploading", fraction }
              : current.kind === "analysing"
                ? current
                : { kind: "analysing", since: Date.now() },
          ),
        id,
      );
      setProgressId(null);
      setPhase({ kind: "idle" });
      onDone(name);
    } catch (error) {
      setProgressId(null);
      setPhase({
        kind: "error",
        message: error instanceof ApiError ? error.message : "Something went wrong",
      });
    }
  }

  function runFile(file: File) {
    if (busy) return;
    if (config && file.size > config.max_upload_mb * MB) {
      setPhase({
        kind: "error",
        message: `Upload too large; the limit is ${config.max_upload_mb} MB`,
      });
      return;
    }
    const body = new FormData();
    body.append("file", file);
    void run("/upload", body);
  }

  function runUrl(url: string) {
    if (busy || !url.trim()) return;
    void run("/download_yt", JSON.stringify({ url: url.trim() }));
  }

  return { phase, busy, server, runFile, runUrl, clearError: () => setPhase({ kind: "idle" }) };
}
