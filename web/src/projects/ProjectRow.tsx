import { useState } from "react";
import { Link, useLocation } from "react-router";
import type { ProjectItem } from "../api";
import { Icon } from "../components/icons";
import RenameForm from "../components/RenameForm";
import { StatusChip } from "../components/StatusChip";

export const COLUMNS = 9;

export function formatDuration(seconds: number | null): string {
  if (seconds === null) return "–";
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// The date part of the UTC stamp: stable across locales and short enough for a compact row.
const formatCreated = (created: string | null) => (created ? created.slice(0, 10) : "–");

export default function ProjectRow({
  item,
  selected,
  onSelect,
  onRename,
  onDelete,
}: {
  item: ProjectItem;
  selected: boolean;
  onSelect: (name: string, picked: boolean) => void;
  onRename: (name: string, to: string) => Promise<boolean>;
  onDelete: (name: string) => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const { name } = item;
  // The page's "Projects" link returns to the list as the person left it, filters included; the
  // facts the events response lacks (duration, created, size) travel along for its subtitle.
  const { search } = useLocation();

  async function save(to: string) {
    setBusy(true);
    const ok = await onRename(name, to);
    setBusy(false);
    if (ok) setRenaming(false);
  }

  if (renaming) {
    return (
      <tr className="prow renaming">
        <td colSpan={COLUMNS}>
          <RenameForm name={name} busy={busy} onSave={save} onCancel={() => setRenaming(false)} />
        </td>
      </tr>
    );
  }
  return (
    <tr className={selected ? "prow picked" : "prow"}>
      <td className="cell-pick">
        <input
          type="checkbox"
          aria-label={`Select ${name}`}
          checked={selected}
          onChange={(event) => onSelect(name, event.target.checked)}
        />
      </td>
      <th scope="row" className="cell-name">
        <Link
          className="name-link"
          to={`/p/${encodeURIComponent(name)}`}
          state={{ list: search, duration: item.duration, created: item.created, size: item.size }}
        >
          {name}
        </Link>
      </th>
      <td className="cell-key">{item.key ?? "–"}</td>
      <td className="cell-bpm num">{item.bpm === null ? "–" : `${item.bpm} BPM`}</td>
      <td className="cell-duration num">{formatDuration(item.duration)}</td>
      <td className="cell-created num">{formatCreated(item.created)}</td>
      <td className="cell-size num">{formatSize(item.size)}</td>
      <td className="cell-status">
        <StatusChip done={item.done} />
      </td>
      <td className="cell-actions">
        <div className="actions-row">
          {item.done && (
            <a
              className="btn icon"
              href={`/download/${encodeURIComponent(name)}`}
              download
              aria-label={`Download ZIP of ${name}`}
            >
              <Icon name="download" />
            </a>
          )}
          <button
            type="button"
            className="btn icon"
            aria-label={`Rename ${name}`}
            onClick={() => setRenaming(true)}
          >
            <Icon name="pencil" />
          </button>
          <button
            type="button"
            className="btn icon"
            aria-label={`Delete ${name}`}
            onClick={() => onDelete(name)}
          >
            <Icon name="trash" />
          </button>
        </div>
      </td>
    </tr>
  );
}
