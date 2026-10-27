import { useEffect, useId, useState, type DragEvent, type FormEvent } from "react";
import { ApiError, postAnalysis, type Config } from "../api";
import Banner from "./Banner";

const MB = 1024 * 1024;

type Tab = "file" | "youtube";
type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; fraction: number }
  | { kind: "analysing"; since: number }
  | { kind: "error"; message: string };

export function sizeLabel(bytes: number): string {
  return bytes >= MB
    ? `${(bytes / MB).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(since);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const seconds = Math.max(0, Math.round((now - since) / 1000));
  return (
    <span className="num">
      {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
    </span>
  );
}

export default function AnalyseCard({
  config,
  onDone,
}: {
  config: Config | null;
  onDone: (name: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputId = useId();
  const busy = phase.kind === "uploading" || phase.kind === "analysing";
  const accept = config?.extensions.join(",") ?? "";

  function choose(chosen: File | undefined) {
    if (!chosen) return;
    setFile(chosen);
    setPhase({ kind: "idle" });
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    let body: FormData | string;
    let endpoint: "/upload" | "/download_yt";
    if (tab === "file") {
      if (!file) return;
      if (config && file.size > config.max_upload_mb * MB) {
        setPhase({
          kind: "error",
          message: `Upload too large; the limit is ${config.max_upload_mb} MB`,
        });
        return;
      }
      body = new FormData();
      body.append("file", file);
      endpoint = "/upload";
      setPhase({ kind: "uploading", fraction: 0 });
    } else {
      if (!url.trim()) return;
      body = JSON.stringify({ url: url.trim() });
      endpoint = "/download_yt";
      setPhase({ kind: "analysing", since: Date.now() });
    }
    try {
      const { name } = await postAnalysis(endpoint, body, (fraction) =>
        setPhase((current) =>
          fraction < 1
            ? { kind: "uploading", fraction }
            : current.kind === "analysing"
              ? current
              : { kind: "analysing", since: Date.now() },
        ),
      );
      setPhase({ kind: "idle" });
      onDone(name);
    } catch (error) {
      setPhase({
        kind: "error",
        message: error instanceof ApiError ? error.message : "Something went wrong",
      });
    }
  }

  return (
    <form className="card" onSubmit={submit} aria-busy={busy}>
      <div className="tabs" role="tablist">
        {(["file", "youtube"] as const).map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={tab === name}
            className={tab === name ? "tab on" : "tab"}
            disabled={busy}
            onClick={() => {
              setTab(name);
              setPhase({ kind: "idle" });
            }}
          >
            {name === "file" ? "File" : "YouTube"}
          </button>
        ))}
      </div>

      {tab === "file" ? (
        <label
          htmlFor={inputId}
          className={dragging ? "drop on" : "drop"}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            id={inputId}
            type="file"
            accept={accept}
            disabled={busy}
            aria-label="Audio file"
            onChange={(event) => choose(event.target.files?.[0])}
          />
          {file ? (
            <span>
              <strong>{file.name}</strong> <span className="muted num">{sizeLabel(file.size)}</span>
            </span>
          ) : (
            <span>
              Drop audio here or <u>choose a file</u>
              <br />
              <span className="muted">{accept.replaceAll(",", " ")}</span>
            </span>
          )}
        </label>
      ) : (
        <input
          className="field"
          type="url"
          inputMode="url"
          placeholder="https://youtu.be/..."
          aria-label="YouTube URL"
          value={url}
          disabled={busy}
          onChange={(event) => setUrl(event.target.value)}
        />
      )}

      {phase.kind === "uploading" && (
        <progress className="bar" max={1} value={phase.fraction} aria-label="Upload progress" />
      )}
      {phase.kind === "analysing" && (
        <p className="status">
          <span className="pulse" aria-hidden="true" />
          Analysing <Elapsed since={phase.since} />
        </p>
      )}
      {phase.kind === "error" && <Banner message={phase.message} />}

      <button
        className="primary"
        type="submit"
        disabled={busy || (tab === "file" ? !file : !url.trim())}
      >
        Analyse
      </button>
    </form>
  );
}
