import { Link } from "react-router";
import type { Project } from "../api";
import Banner from "./Banner";
import { DownloadIcon } from "./icons";

export default function ProjectList({
  projects,
  error,
  loading,
}: {
  projects: Project[] | null;
  error: string | null;
  loading: boolean;
}) {
  return (
    <section className="card" aria-labelledby="projects-title">
      <h2 id="projects-title">Projects</h2>
      {error && <Banner message={error} />}
      {loading && !projects && <p className="muted">Loading…</p>}
      {projects?.length === 0 && <p className="muted">No analyses yet.</p>}
      {projects && projects.length > 0 && (
        <ul className="rows">
          {projects.map((project) => (
            <li key={project.name} className="row">
              <Link className="row-main" to={`/p/${encodeURIComponent(project.name)}`}>
                <span className="name">{project.name}</span>
                <span className="facts num">
                  {project.meta?.key ?? "?"} ·{" "}
                  {project.meta?.bpm ? `${project.meta.bpm} BPM` : "? BPM"}
                </span>
              </Link>
              {project.done ? (
                <a className="chip" href={`/download/${encodeURIComponent(project.name)}`} download>
                  <DownloadIcon /> ZIP
                </a>
              ) : (
                <span className="chip off">not built</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
