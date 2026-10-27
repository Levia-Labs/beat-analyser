import { useCallback, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { ApiError, deleteProject, getEvents, renameProject } from "../api";
import Banner from "../components/Banner";
import ConfirmDialog from "../components/ConfirmDialog";
import { Icon } from "../components/icons";
import RenameForm from "../components/RenameForm";
import Timeline from "../components/Timeline";
import { message, useLoad } from "../useLoad";

export function ViewerRoute() {
  const { name = "" } = useParams();
  return <Viewer key={name} name={name} />;
}

export default function Viewer({ name }: { name: string }) {
  const load = useCallback(() => getEvents(name), [name]);
  const events = useLoad(load);
  const navigate = useNavigate();
  const [mode, setMode] = useState<"view" | "rename" | "delete">("view");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const renameButton = useRef<HTMLButtonElement>(null);
  const deleteButton = useRef<HTMLButtonElement>(null);

  // Every mode change drops the old error; closing returns focus to the button that opened it.
  function open(next: "rename" | "delete") {
    setProblem(null);
    setMode(next);
  }
  function close() {
    const opener = mode === "rename" ? renameButton : deleteButton;
    setProblem(null);
    setMode("view");
    opener.current?.focus();
  }

  async function act(action: () => Promise<void>) {
    setBusy(true);
    setProblem(null);
    try {
      await action();
    } catch (error) {
      setProblem(message(error));
      setBusy(false);
    }
  }
  const remove = () =>
    act(async () => {
      await deleteProject(name);
      navigate("/");
    });
  const rename = (to: string) =>
    act(async () => {
      const { name: renamed } = await renameProject(name, to);
      navigate(`/p/${encodeURIComponent(renamed)}`, { replace: true });
    });

  if (events.error instanceof ApiError && events.error.status === 404) {
    return (
      <section className="card">
        <p>No analysed project called {name}</p>
        <p className="muted">It may still be analysing.</p>
        <div className="head">
          <button type="button" className="btn" onClick={events.reload}>
            <Icon name="refresh" /> Reload
          </button>
          <Link className="btn" to="/">
            <Icon name="list" /> Home
          </Link>
        </div>
      </section>
    );
  }
  if (events.error) {
    return (
      <Banner
        message={message(events.error) ?? "Something went wrong"}
        action={
          <button type="button" className="btn" onClick={events.reload}>
            <Icon name="refresh" /> Retry
          </button>
        }
      />
    );
  }
  if (!events.data) return <p className="muted">Loading…</p>;

  const { meta } = events.data;
  return (
    <>
      <section className="card head">
        <Link to="/" className="btn plain icon" aria-label="Back to projects">
          <Icon name="arrow-left" />
        </Link>
        <h1>{name}</h1>
        <span className="muted num">
          {meta?.key ?? "?"} · {meta?.bpm ? `${meta.bpm} BPM` : "? BPM"}
        </span>
        <a className="btn" href={`/download/${encodeURIComponent(name)}`} download>
          <Icon name="download" /> ZIP
        </a>
        <button
          ref={renameButton}
          type="button"
          className="btn"
          onClick={() => open("rename")}
          disabled={busy}
        >
          <Icon name="pencil" /> Rename
        </button>
        <button
          ref={deleteButton}
          type="button"
          className="btn"
          onClick={() => open("delete")}
          disabled={busy}
        >
          <Icon name="trash" /> Delete
        </button>
      </section>
      {problem && <Banner message={problem} />}
      {mode === "rename" && (
        <section className="card">
          <RenameForm name={name} busy={busy} onSave={rename} onCancel={close} />
        </section>
      )}
      {mode === "delete" && (
        <ConfirmDialog
          title={`Delete ${name}?`}
          body="Audio, CSVs and ZIP are removed."
          confirm={`Delete ${name}`}
          busy={busy}
          onConfirm={remove}
          onCancel={close}
        />
      )}
      <Timeline events={events.data} />
    </>
  );
}
