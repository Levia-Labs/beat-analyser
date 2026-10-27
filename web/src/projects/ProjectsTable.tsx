import { useEffect, useRef } from "react";
import type { ProjectItem } from "../api";
import type { FilterProps } from "./ColumnFilters";
import ProjectRow, { COLUMNS } from "./ProjectRow";
import ProjectsHead from "./ProjectsHead";
import type { SortKey } from "./query";

export default function ProjectsTable({
  items,
  filters,
  loading,
  loadingMore,
  hasMore,
  emptyText,
  selected,
  onSort,
  onSelect,
  onSelectAll,
  onNearEnd,
  onRename,
  onDelete,
}: {
  items: ProjectItem[];
  filters: FilterProps;
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  emptyText: string | null;
  selected: Set<string>;
  onSort: (column: SortKey) => void;
  onSelect: (name: string, picked: boolean) => void;
  onSelectAll: (picked: boolean) => void;
  onNearEnd: () => void;
  onRename: (name: string, to: string) => Promise<boolean>;
  onDelete: (name: string) => void;
}) {
  const region = useRef<HTMLDivElement>(null);
  const sentinel = useRef<HTMLDivElement>(null);

  // The observer is rebuilt whenever onNearEnd changes (after each page), and a new observer
  // reports the sentinel's state at once: a short page that leaves it in view loads the next one.
  useEffect(() => {
    const target = sentinel.current;
    if (!hasMore || !target || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) onNearEnd();
      },
      { root: region.current, rootMargin: "0px 0px 200px 0px" },
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [hasMore, onNearEnd]);

  const allPicked = items.length > 0 && items.every((item) => selected.has(item.name));
  return (
    <div className="table-region" ref={region}>
      <table className="ptable">
        <ProjectsHead
          props={filters}
          allPicked={allPicked}
          onSelectAll={onSelectAll}
          onSort={onSort}
        />
        <tbody>
          {items.map((item) => (
            <ProjectRow
              key={item.name}
              item={item}
              selected={selected.has(item.name)}
              onSelect={onSelect}
              onRename={onRename}
              onDelete={onDelete}
            />
          ))}
          {(loading || loadingMore) && (
            <tr className="loading-row">
              <td colSpan={COLUMNS} className="muted">
                Loading…
              </td>
            </tr>
          )}
          {!loading && items.length === 0 && emptyText && (
            <tr className="empty-row">
              <td colSpan={COLUMNS} className="muted">
                {emptyText}
              </td>
            </tr>
          )}
        </tbody>
      </table>
      <div ref={sentinel} className="sentinel" />
    </div>
  );
}
