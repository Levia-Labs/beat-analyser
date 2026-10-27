import { useCallback, useState } from "react";
import { ApiError, deleteProject, renameProject } from "../api";
import Banner from "../components/Banner";
import ConfirmDialog from "../components/ConfirmDialog";
import { Icon } from "../components/icons";
import { message } from "../useLoad";
import { hasFilters } from "./query";
import ProjectsTable from "./ProjectsTable";
import ProjectsBar from "./ProjectsBar";
import { useDraftText } from "./useDraftText";
import { usePagedProjects } from "./usePagedProjects";
import { useProjectsQuery } from "./useProjectsQuery";

export default function ProjectsScreen() {
  const { query, update, sortBy, clear } = useProjectsQuery();
  const writeName = useCallback((q: string, replace: boolean) => update({ q }, replace), [update]);
  const nameDraft = useDraftText(query.q, writeName);
  const filters = { query, update, nameDraft };
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
    let stoppedAt = -1;
    for (const [index, name] of confirming.entries()) {
      try {
        await deleteProject(name);
      } catch (error) {
        failures.push(`${name}: ${message(error)}`);
        // Each refused DELETE counts in the server's failure limiter; going on would lock the IP out.
        if (error instanceof ApiError && (error.status === 401 || error.status === 429)) {
          stoppedAt = index + 1;
          break;
        }
      }
    }
    if (stoppedAt >= 0 && stoppedAt < confirming.length) {
      failures.push(`not attempted: ${confirming.slice(stoppedAt).join(", ")}`);
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
      <ProjectsBar
        props={filters}
        total={paged.loading ? null : paged.total}
        onSort={sortBy}
        onClear={() => {
          nameDraft.reset();
          clear();
        }}
      />
      {paged.error && (
        <Banner
          message={paged.error}
          action={
            <button type="button" className="btn" onClick={paged.retry}>
              <Icon name="refresh" /> Retry
            </button>
          }
        />
      )}
      {problem && <Banner message={problem} />}
      {selected.size > 0 && (
        <div className="bulk">
          <button
            type="button"
            className="btn primary"
            onClick={() => setConfirming([...selected])}
          >
            <Icon name="trash" /> Delete selected ({selected.size})
          </button>
        </div>
      )}
      <ProjectsTable
        items={items}
        filters={filters}
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
