import { Link, Route, Routes } from "react-router";
import Logo from "./components/Logo";

export default function App() {
  return (
    <>
      <header className="top">
        <Link to="/" className="brand">
          <Logo />
          <span>Beat Analyser</span>
        </Link>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<section className="card">Home</section>} />
          <Route path="/p/:name" element={<section className="card">Viewer</section>} />
          <Route
            path="*"
            element={
              <p className="muted">
                Nothing here. <Link to="/">Home</Link>
              </p>
            }
          />
        </Routes>
      </main>
    </>
  );
}
