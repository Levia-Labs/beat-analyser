import { Icon } from "../components/icons";
import { BackLink } from "../components/ProjectHead";
import { useDocumentTitle } from "../useDocumentTitle";

const GITHUB = "https://github.com/Levia-Labs/beat-analyser";
const LEVIA = "https://levialabs.net";
const MCP_DOC = `${GITHUB}/blob/main/docs/mcp.md`;

function External({ href, children }: { href: string; children: string }) {
  return (
    <a className="btn" href={href} target="_blank" rel="noopener noreferrer">
      <Icon name="external-link" /> {children}
    </a>
  );
}

export default function About() {
  useDocumentTitle("About · Beat Analyser");
  const commit = import.meta.env.VITE_COMMIT || "dev";
  return (
    <section className="about">
      <BackLink />
      <h1>About</h1>
      <h2>What it is</h2>
      <p className="muted">
        Beat Analyser listens to a song and writes down when every drum hit, beat and note happens,
        so you can use the timing in games, visuals or your own tools.
      </p>
      <h2>How it works</h2>
      <ol className="muted" aria-label="Steps">
        <li>Add a song: a file or a YouTube link.</li>
        <li>It finds the tempo, beats, kicks, snares, hi-hats, bass, chords and melody.</li>
        <li>Browse the timeline and download a ZIP of CSV files.</li>
      </ol>
      <h2>What you get</h2>
      <p className="muted">One CSV per track (time, label), plus the key and BPM.</p>
      <h2>Runs on your machine</h2>
      <p className="muted">
        Audio stays in your uploads/ folder. Nothing leaves it except YouTube downloads.
      </p>
      <h2>For agents</h2>
      <p className="muted">
        An MCP endpoint at <code>http://localhost:8001/mcp</code> lets agents analyse songs; see{" "}
        <a href={MCP_DOC} target="_blank" rel="noopener noreferrer">
          docs/mcp.md
        </a>{" "}
        on GitHub.
      </p>
      <h2>Links</h2>
      <div className="about-links">
        <External href={GITHUB}>GitHub</External>
        <External href={LEVIA}>Levia Labs</External>
      </div>
      <p className="muted about-version">Version {commit}</p>
    </section>
  );
}
