import { useCallback } from "react";
import { Link, useParams } from "react-router";
import { ApiError, getEvents } from "../api";
import Banner from "../components/Banner";
import Timeline from "../components/Timeline";
import { message, useLoad } from "../useLoad";

export function ViewerRoute() {
  const { name = "" } = useParams();
  return <Viewer key={name} name={name} />;
}

export default function Viewer({ name }: { name: string }) {
  const load = useCallback(() => getEvents(name), [name]);
  const events = useLoad(load);

  if (events.error instanceof ApiError && events.error.status === 404) {
    return (
      <section className="card">
        <p>No analysed project called {name}</p>
        <p className="muted">It may still be analysing.</p>
        <div className="head">
          <button type="button" className="chip" onClick={events.reload}>
            Reload
          </button>
          <Link className="chip" to="/">
            Home
          </Link>
        </div>
      </section>
    );
  }
  if (events.error) {
    return (
      <Banner
        message={message(events.error) ?? "Something went wrong"}
        action={
          <button type="button" className="chip" onClick={events.reload}>
            Retry
          </button>
        }
      />
    );
  }
  if (!events.data) return <p className="muted">Loading…</p>;

  const { meta } = events.data;
  return (
    <>
      <section className="card head">
        <Link to="/" aria-label="Back to projects">
          ←
        </Link>
        <h1>{name}</h1>
        <span className="muted num">
          {meta?.key ?? "?"} · {meta?.bpm ? `${meta.bpm} BPM` : "? BPM"}
        </span>
        <a className="chip" href={`/download/${encodeURIComponent(name)}`} download>
          ZIP
        </a>
      </section>
      <Timeline events={events.data} />
    </>
  );
}
