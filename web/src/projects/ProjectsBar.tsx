import AddModal from "../components/AddModal";
import { Icon } from "../components/icons";
import type { FilterProps } from "./ColumnFilters";
import MobileFilters from "./MobileFilters";
import { totalFilters, type SortKey } from "./query";

// The list's header bar: title and count on the left; Clear filters (only with filters), the phone's
// Filters button and Add on the right.
export default function ProjectsBar({
  props,
  total,
  onSort,
  onClear,
}: {
  props: FilterProps;
  total: number | null;
  onSort: (column: SortKey) => void;
  onClear: () => void;
}) {
  const active = totalFilters(props.query);
  return (
    <div className="listbar">
      <div className="listbar-title">
        <h2 id="projects-title">Projects</h2>
        {total !== null && (
          <span
            className="count num"
            role="status"
            aria-label={`${total} ${total === 1 ? "project" : "projects"}`}
          >
            {total}
          </span>
        )}
      </div>
      <div className="listbar-actions">
        {active > 0 && (
          <button type="button" className="btn" onClick={onClear}>
            <Icon name="filter-x" /> Clear filters ({active})
          </button>
        )}
        <MobileFilters props={props} total={total} onSort={onSort} onClear={onClear} />
        <AddModal />
      </div>
    </div>
  );
}
