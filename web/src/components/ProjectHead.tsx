import { Link, useLocation } from "react-router";
import type { Events } from "../api";
import { formatDuration, formatSize } from "../projects/ProjectRow";
import { Icon } from "./icons";
import { StatusChip } from "./StatusChip";

// The list's query, passed in the link's state; absent when the page was opened directly.
function listQuery(state: unknown): string {
  const list = state && typeof state === "object" ? (state as { list?: unknown }).list : "";
  return typeof list === "string" ? list : "";
}

export function BackLink() {
  const list = listQuery(useLocation().state);
  return (
    <Link className="back" to={`/${list}`}>
      <Icon name="arrow-left" />
      Projects
    </Link>
  );
}

// Name, the facts known without another request, then status, key and tempo chips.
export function ProjectHead({ events }: { events: Events }) {
  const { name, meta, duration, created, size } = events;
  const sub = [
    typeof duration === "number" ? formatDuration(duration) : null,
    created ? created.slice(0, 10) : null,
    typeof size === "number" ? formatSize(size) : null,
  ].filter(Boolean);
  return (
    <div className="maphead">
      <div className="mapmeta">
        <h1>{name}</h1>
        {sub.length > 0 && <p className="sub">{sub.join(" · ")}</p>}
        <div className="metarow">
          <StatusChip done />
          <span className="chip key-chip">{meta?.key ?? "?"}</span>
          <span className="tempo">{meta?.bpm ? `${meta.bpm} BPM` : "? BPM"}</span>
        </div>
      </div>
    </div>
  );
}
