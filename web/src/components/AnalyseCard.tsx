import { useEffect, useId, useState, type DragEvent, type FormEvent } from "react";
import type { Config } from "../api";
import Banner from "./Banner";
import { Icon } from "./icons";
import StageList from "./StageList";
import { sizeLabel, useAnalysis, type Phase } from "./useAnalysis";

export { sizeLabel };

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

function UrlPart({
  busy,
  onRun,
  onEdit,
}: {
  busy: boolean;
  onRun: (url: string) => void;
  onEdit: () => void;
}) {
  const [url, setUrl] = useState("");
  function submit(event: FormEvent) {
    event.preventDefault();
    onRun(url);
  }
  return (
    <form className="analyse-part" onSubmit={submit}>
      <input
        className="field"
        type="url"
        inputMode="url"
        placeholder="https://youtu.be/..."
        aria-label="YouTube URL"
        value={url}
        disabled={busy}
        onChange={(event) => {
          setUrl(event.target.value);
          onEdit();
        }}
      />
      <button
        className="btn primary"
        type="submit"
        aria-label="Analyse YouTube URL"
        disabled={busy || !url.trim()}
      >
        <Icon name="play" /> Analyse
      </button>
    </form>
  );
}

function FilePart({
  config,
  busy,
  onRun,
  onEdit,
}: {
  config: Config | null;
  busy: boolean;
  onRun: (file: File) => void;
  onEdit: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputId = useId();
  const accept = config?.extensions.join(",") ?? "";

  function choose(chosen: File | undefined) {
    if (!chosen || busy) return;
    setFile(chosen);
    onEdit();
  }
  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0]);
  }
  function submit(event: FormEvent) {
    event.preventDefault();
    if (file) onRun(file);
  }
  return (
    <form className="analyse-part" onSubmit={submit}>
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
            <Icon name="upload" /> Drop audio here or <u>choose a file</u>
            <br />
            <span className="muted">{accept.replaceAll(",", " ")}</span>
          </span>
        )}
      </label>
      <button
        className="btn primary"
        type="submit"
        aria-label="Analyse file"
        disabled={busy || !file}
      >
        <Icon name="play" /> Analyse
      </button>
    </form>
  );
}

function RunStatus({ phase }: { phase: Phase }) {
  return (
    <>
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
    </>
  );
}

// The two ways in, one run at a time: a YouTube URL first, then a local file.
export default function AnalyseCard({
  config,
  onDone,
  onBusy,
}: {
  config: Config | null;
  onDone: (name: string) => void;
  onBusy?: (busy: boolean) => void;
}) {
  const { phase, busy, server, runFile, runUrl, clearError } = useAnalysis(config, onDone);
  useEffect(() => onBusy?.(busy), [busy, onBusy]);
  const edit = () => phase.kind === "error" && clearError();
  return (
    <div className="analyse" aria-busy={busy}>
      <UrlPart busy={busy} onRun={runUrl} onEdit={edit} />
      <div className="or" aria-hidden="true">
        <span>or</span>
      </div>
      <FilePart config={config} busy={busy} onRun={runFile} onEdit={edit} />
      <RunStatus phase={phase} />
      {busy && server && <StageList progress={server} />}
    </div>
  );
}
