import * as Dialog from "@radix-ui/react-dialog";
import { Icon } from "../components/icons";
import { ColumnFilter, FILTER_COLUMNS, type FilterProps } from "./ColumnFilters";
import { activeCounts, SORT_KEYS, SORT_LABELS, totalFilters, type SortKey } from "./query";

// The phone's one Filters button: every column's filter in a sheet, plus the sort controls the
// hidden column headers would have given.
export default function MobileFilters({
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
  const { query } = props;
  const counts = activeCounts(query);
  const active = totalFilters(query);
  const order = query.order === "asc" ? "sort-asc" : "sort-desc";
  const show =
    total === null ? "Show projects" : `Show ${total} ${total === 1 ? "project" : "projects"}`;
  return (
    <Dialog.Root>
      <Dialog.Trigger
        className="btn filters-button"
        aria-label={active > 0 ? `Filters, ${active} active` : "Filters"}
      >
        <Icon name="filter" /> Filters
        {active > 0 && (
          <span className="filter-badge" aria-hidden="true">
            {active}
          </span>
        )}
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="overlay" />
        <Dialog.Content className="sheet" aria-describedby={undefined}>
          <div className="modal-head">
            <Dialog.Title className="modal-title">Filters</Dialog.Title>
            <Dialog.Close className="btn plain icon" aria-label="Close filters">
              <Icon name="x" />
            </Dialog.Close>
          </div>
          <div className="sheet-body">
            {FILTER_COLUMNS.map(({ column, label }) => (
              <section key={column} className="acc">
                <h3>
                  {label}
                  {counts[column] > 0 && <span className="filter-badge">{counts[column]}</span>}
                </h3>
                <ColumnFilter column={column} {...props} />
              </section>
            ))}
            <section className="acc">
              <h3>Sort</h3>
              <div className="filter-row">
                <select
                  className="field"
                  aria-label="Sort"
                  value={query.sort}
                  onChange={(event) => {
                    // Picking the current column again would flip the order; the select only switches.
                    const column = event.target.value as SortKey;
                    if (column !== query.sort) onSort(column);
                  }}
                >
                  {SORT_KEYS.map((column) => (
                    <option key={column} value={column}>
                      {SORT_LABELS[column]}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn icon"
                  aria-label={query.order === "asc" ? "Ascending, reverse" : "Descending, reverse"}
                  onClick={() => onSort(query.sort)}
                >
                  <Icon name={order} />
                </button>
              </div>
            </section>
          </div>
          <div className="sheet-foot">
            <button
              type="button"
              className="btn icon"
              aria-label="Clear all"
              title="Clear all"
              disabled={active === 0}
              onClick={onClear}
            >
              <Icon name="filter-x" />
            </button>
            <Dialog.Close className="btn primary">
              <Icon name="check" /> {show}
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
