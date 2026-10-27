import { ChecklistBody, RangeFilterBody, TextFilterBody, type Draft } from "./FilterBodies";
import { KEY_OPTIONS, keyList, withKeys, type FilterColumn, type ProjectsQuery } from "./query";

// The five column filters, as bodies: the header popovers and the phone sheet render the same ones.
export type Update = (patch: Partial<ProjectsQuery>, replace?: boolean) => void;

export type FilterProps = { query: ProjectsQuery; update: Update; nameDraft: Draft };

export const FILTER_COLUMNS: { column: FilterColumn; label: string }[] = [
  { column: "name", label: "Name" },
  { column: "key", label: "Key" },
  { column: "bpm", label: "BPM" },
  { column: "duration", label: "Duration" },
  { column: "status", label: "Status" },
];

const KEY_CHOICES = KEY_OPTIONS.map((value) => ({
  value,
  label: value === "major" || value === "minor" ? value[0]!.toUpperCase() + value.slice(1) : value,
}));

const STATUS_CHOICES = [
  { value: "built", label: "Built" },
  { value: "pending", label: "Not built" },
];

export function ColumnFilter({
  column,
  query,
  update,
  nameDraft,
}: FilterProps & { column: FilterColumn }) {
  switch (column) {
    case "name":
      return <TextFilterBody draft={nameDraft} active={nameDraft.value !== "" || query.q !== ""} />;
    case "key":
      return (
        <ChecklistBody
          group="Key"
          options={KEY_CHOICES}
          selected={keyList(query.key)}
          onChange={(next) => update({ key: withKeys(next) })}
          onClear={() => update({ key: "" })}
        />
      );
    case "bpm":
      return (
        <RangeFilterBody
          name="BPM"
          min={query.bpmMin}
          max={query.bpmMax}
          onChange={(range, replace) => update({ bpmMin: range.min, bpmMax: range.max }, replace)}
          onClear={() => update({ bpmMin: "", bpmMax: "" })}
        />
      );
    case "duration":
      return (
        <RangeFilterBody
          name="Duration"
          unit=" (s)"
          min={query.durMin}
          max={query.durMax}
          onChange={(range, replace) => update({ durMin: range.min, durMax: range.max }, replace)}
          onClear={() => update({ durMin: "", durMax: "" })}
        />
      );
    case "status":
      return (
        <ChecklistBody
          group="Status"
          options={STATUS_CHOICES}
          selected={query.status ? [query.status] : []}
          // Both choices, like none, mean any status.
          onChange={(next) =>
            update({ status: next.length === 1 ? (next[0] === "built" ? "built" : "pending") : "" })
          }
          onClear={() => update({ status: "" })}
        />
      );
  }
}
