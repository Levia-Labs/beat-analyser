import type { ReactNode } from "react";
import { Icon } from "../components/icons";
import { ColumnFilter, type FilterProps } from "./ColumnFilters";
import FilterPopover from "./FilterPopover";
import { activeCounts, SORT_LABELS, type FilterColumn, type SortKey } from "./query";

type Cell = { label: string; sort?: SortKey; filter?: FilterColumn };

const CELLS: Cell[] = [
  { label: SORT_LABELS.name, sort: "name", filter: "name" },
  { label: SORT_LABELS.key, sort: "key", filter: "key" },
  { label: SORT_LABELS.bpm, sort: "bpm", filter: "bpm" },
  { label: SORT_LABELS.duration, sort: "duration", filter: "duration" },
  { label: SORT_LABELS.created, sort: "created" },
  { label: SORT_LABELS.size, sort: "size" },
  { label: "Status", filter: "status" },
];

function HeaderCell({
  cell,
  props,
  onSort,
  count,
}: {
  cell: Cell;
  props: FilterProps;
  onSort: (column: SortKey) => void;
  count: number;
}) {
  const { query } = props;
  const { label, sort, filter } = cell;
  const active = sort !== undefined && query.sort === sort;
  const ascending = query.order === "asc";
  const direction = ascending ? "sort-asc" : "sort-desc";
  let order: "ascending" | "descending" | "none" = "none";
  if (active) order = ascending ? "ascending" : "descending";
  let body: ReactNode = <span className="th-label">{label}</span>;
  if (sort) {
    body = (
      <button
        type="button"
        className={active ? "sort-button on" : "sort-button"}
        aria-label={`Sort by ${label}`}
        onClick={() => onSort(sort)}
      >
        {label} <Icon name={direction} />
      </button>
    );
  }
  return (
    <th scope="col" className={`col-${sort ?? filter}`} aria-sort={sort ? order : undefined}>
      <div className="th-row">
        {body}
        {filter && (
          <FilterPopover label={label} count={count}>
            <ColumnFilter column={filter} {...props} />
          </FilterPopover>
        )}
      </div>
    </th>
  );
}

export default function ProjectsHead({
  props,
  allPicked,
  onSelectAll,
  onSort,
}: {
  props: FilterProps;
  allPicked: boolean;
  onSelectAll: (picked: boolean) => void;
  onSort: (column: SortKey) => void;
}) {
  const counts = activeCounts(props.query);
  return (
    <thead>
      <tr>
        <th scope="col" className="cell-pick">
          <input
            type="checkbox"
            aria-label="Select all loaded"
            checked={allPicked}
            onChange={(event) => onSelectAll(event.target.checked)}
          />
        </th>
        {CELLS.map((cell) => (
          <HeaderCell
            key={cell.label}
            cell={cell}
            props={props}
            onSort={onSort}
            count={cell.filter ? counts[cell.filter] : 0}
          />
        ))}
        <th scope="col" className="col-actions">
          <span className="sr-only">Actions</span>
        </th>
      </tr>
    </thead>
  );
}
