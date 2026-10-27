import { useCallback, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { ApiError, deleteProject, getEvents, renameProject } from "../api";
import Banner from "../components/Banner";
import ConfirmDialog from "../components/ConfirmDialog";
import { Icon } from "../components/icons";
import { BackLink, ProjectHead } from "../components/ProjectHead";
import RenameForm from "../components/RenameForm";
import Timeline from "../components/Timeline";
import { useDocumentTitle } from "../useDocumentTitle";
import { message, useLoad } from "../useLoad";

export function ViewerRoute() {
  const { name = "" } = useParams();
  return <Viewer key={name} name={name} />;
}

export default function Viewer({ name }: { name: string }) {
  const load = useCallback(() => getEvents(name), [name]);
  const events = useLoad(load);
  const navigate = useNavigate();
  // Only a loaded project names the tab; the not-found and error states keep the plain title.
  useDocumentTitle(events.data ? `${name} · Beat Analyser` : "Beat Analyser");
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
      <section className="page map-page">
        <BackLink />
        <div className="banner info">
          <div>
            <p>No analysed project called {name}</p>
            <p className="muted">It may still be analysing.</p>
          </div>
          <div className="map-toolbar">
            <button type="button" className="btn" onClick={events.reload}>
              <Icon name="refresh" /> Reload
            </button>
            <Link className="btn" to="/">
              <Icon name="list" /> Home
            </Link>
          </div>
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

  return (
    <section className="page map-page">
      <BackLink />
      <ProjectHead events={events.data} />
      <div className="map-toolbar">
        <span className="spacer" />
        <a className="btn primary" href={`/download/${encodeURIComponent(name)}`} download>
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
      </div>
      {problem && <Banner message={problem} />}
      {mode === "rename" && (
        <div className="rename-panel">
          <RenameForm name={name} busy={busy} onSave={rename} onCancel={close} />
        </div>
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
      <h2 className="group-title">Timeline</h2>
      <Timeline events={events.data} />
    </section>
  );
}
