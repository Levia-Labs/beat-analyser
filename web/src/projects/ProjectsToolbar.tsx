import { useEffect, useState } from "react";
import { FilterIcon, SearchIcon, SortAscIcon, SortDescIcon, XIcon } from "../components/icons";
import { SORT_KEYS, type ProjectsQuery, type SortKey, type Status } from "./query";

export const DEBOUNCE_MS = 250;
export const SORT_LABELS: Record<SortKey, string> = {
  name: "Name",
  key: "Key",
  bpm: "BPM",
  duration: "Duration",
  created: "Created",
  size: "Size",
};

const NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const KEY_NAMES = ["major", "minor"].flatMap((mode) => NOTES.map((note) => `${note} ${mode}`));

type Draft = { q: string; bpmMin: string; bpmMax: string };
const draftOf = ({ q, bpmMin, bpmMax }: ProjectsQuery): Draft => ({ q, bpmMin, bpmMax });

export default function ProjectsToolbar({
  query,
  total,
  onChange,
  onSort,
  onClear,
}: {
  query: ProjectsQuery;
  total: number | null;
  onChange: (patch: Partial<ProjectsQuery>, replace?: boolean) => void;
  onSort: (column: SortKey) => void;
  onClear: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<Draft>(draftOf(query));
  const [seen, setSeen] = useState(query);

  // The URL is the truth. A field follows it when it changed from outside (Clear, back button);
  // fields the URL did not touch keep what is being typed.
  if (seen !== query) {
    setSeen(query);
    setDraft((now) => ({
      q: seen.q !== query.q ? query.q : now.q,
      bpmMin: seen.bpmMin !== query.bpmMin ? query.bpmMin : now.bpmMin,
      bpmMax: seen.bpmMax !== query.bpmMax ? query.bpmMax : now.bpmMax,
    }));
  }

  useEffect(() => {
    const queryDraft = draftOf(query);
    if (
      draft.q === queryDraft.q &&
      draft.bpmMin === queryDraft.bpmMin &&
      draft.bpmMax === queryDraft.bpmMax
    ) {
      return;
    }
    // replace: typing must not fill the back button with one entry per pause.
    const timer = setTimeout(() => onChange(draft, true), DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [draft, query, onChange]);

  const OrderIcon = query.order === "asc" ? SortAscIcon : SortDescIcon;
  return (
    <div className="toolbar">
      <div className="toolbar-main">
        <label className="search">
          <SearchIcon />
          <input
            type="search"
            className="field"
            aria-label="Search by name"
            placeholder="Search by name"
            value={draft.q}
            onChange={(event) => setDraft({ ...draft, q: event.target.value })}
          />
        </label>
        <button
          type="button"
          className="chip filters-toggle"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          <FilterIcon /> Filters
        </button>
        <span className="count num" aria-live="polite">
          {total === null ? "" : `${total} ${total === 1 ? "project" : "projects"}`}
        </span>
      </div>
      <div className={open ? "filters open" : "filters"} role="group" aria-label="Filters">
        <select
          className="field"
          aria-label="Key"
          value={query.key}
          onChange={(event) => onChange({ key: event.target.value })}
        >
          <option value="">Any key</option>
          <option value="major">Major</option>
          <option value="minor">Minor</option>
          {KEY_NAMES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <input
          type="number"
          className="field bpm"
          aria-label="BPM min"
          placeholder="BPM min"
          min={0}
          value={draft.bpmMin}
          onChange={(event) => setDraft({ ...draft, bpmMin: event.target.value })}
        />
        <input
          type="number"
          className="field bpm"
          aria-label="BPM max"
          placeholder="BPM max"
          min={0}
          value={draft.bpmMax}
          onChange={(event) => setDraft({ ...draft, bpmMax: event.target.value })}
        />
        <select
          className="field"
          aria-label="Status"
          value={query.status}
          onChange={(event) => onChange({ status: event.target.value as Status })}
        >
          <option value="">Any status</option>
          <option value="built">Built</option>
          <option value="pending">Not built</option>
        </select>
        <button
          type="button"
          className="chip"
          onClick={() => {
            // Resetting the draft re-runs the debounce effect, which cancels a pending write of the
            // typed text; relying on the URL change alone misses a URL that was already empty.
            setDraft({ q: "", bpmMin: "", bpmMax: "" });
            onClear();
          }}
        >
          <XIcon /> Clear
        </button>
      </div>
      <div className="sort-phone">
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
          className="chip"
          aria-label={query.order === "asc" ? "Ascending, reverse" : "Descending, reverse"}
          onClick={() => onSort(query.sort)}
        >
          <OrderIcon />
        </button>
      </div>
    </div>
  );
}
