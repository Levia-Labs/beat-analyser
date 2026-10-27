import { useState } from "react";
import { deleteProject, renameProject } from "../api";
import Banner from "../components/Banner";
import ConfirmDialog from "../components/ConfirmDialog";
import { RefreshIcon, TrashIcon } from "../components/icons";
import { message } from "../useLoad";
import { hasFilters } from "./query";
import ProjectsTable from "./ProjectsTable";
import ProjectsToolbar from "./ProjectsToolbar";
import { usePagedProjects } from "./usePagedProjects";
import { useProjectsQuery } from "./useProjectsQuery";

export default function ProjectsScreen() {
  const { query, update, sortBy, clear } = useProjectsQuery();
  const paged = usePagedProjects(query);
  const { items, reload } = paged;
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [confirming, setConfirming] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  // Only rows still loaded count: a filter change must not delete names the person no longer sees.
  const selected = new Set(items.filter((item) => picked.has(item.name)).map((item) => item.name));

  function select(name: string, on: boolean) {
    const next = new Set(selected);
    if (on) next.add(name);
    else next.delete(name);
    setPicked(next);
  }

  async function rename(name: string, to: string): Promise<boolean> {
    setProblem(null);
    try {
      await renameProject(name, to);
    } catch (error) {
      setProblem(`${name}: ${message(error)}`);
      return false;
    }
    reload();
    return true;
  }

  // One DELETE per name, in sequence: a failure names its project and the rest still go.
  async function removeConfirmed() {
    if (!confirming) return;
    setBusy(true);
    setProblem(null);
    const failures: string[] = [];
    for (const name of confirming) {
      try {
        await deleteProject(name);
      } catch (error) {
        failures.push(`${name}: ${message(error)}`);
      }
    }
    setBusy(false);
    setConfirming(null);
    setPicked(new Set());
    setProblem(failures.length > 0 ? failures.join("; ") : null);
    reload();
  }

  const emptyText = paged.error
    ? null
    : hasFilters(query)
      ? "No projects match these filters."
      : "No projects yet.";
  const count = confirming?.length ?? 0;
  return (
    <section className="card projects" aria-labelledby="projects-title">
      <h2 id="projects-title" className="sr-only">
        Projects
      </h2>
      <ProjectsToolbar
        query={query}
        total={paged.loading ? null : paged.total}
        onChange={update}
        onSort={sortBy}
        onClear={clear}
      />
      {paged.error && (
        <Banner
          message={paged.error}
          action={
            <button type="button" className="chip" onClick={paged.retry}>
              <RefreshIcon /> Retry
            </button>
          }
        />
      )}
      {problem && <Banner message={problem} />}
      {selected.size > 0 && (
        <div className="bulk">
          <button type="button" className="primary" onClick={() => setConfirming([...selected])}>
            <TrashIcon /> Delete selected ({selected.size})
          </button>
        </div>
      )}
      <ProjectsTable
        items={items}
        query={query}
        loading={paged.loading}
        loadingMore={paged.loadingMore}
        hasMore={paged.hasMore}
        emptyText={emptyText}
        selected={selected}
        onSort={sortBy}
        onSelect={select}
        onSelectAll={(on) => setPicked(on ? new Set(items.map((item) => item.name)) : new Set())}
        onNearEnd={paged.loadMore}
        onRename={rename}
        onDelete={(name) => setConfirming([name])}
      />
      {confirming && (
        <ConfirmDialog
          title={count === 1 ? `Delete ${confirming[0]}?` : `Delete ${count} projects?`}
          body="Audio, CSVs and ZIP are removed."
          confirm="Delete"
          busy={busy}
          onConfirm={removeConfirmed}
          onCancel={() => setConfirming(null)}
        />
      )}
    </section>
  );
}
