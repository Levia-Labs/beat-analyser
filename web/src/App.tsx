import { Link, Route, Routes } from "react-router";
import Logo from "./components/Logo";
import Home from "./screens/Home";
import { ViewerRoute } from "./screens/Viewer";

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
          <Route path="/" element={<Home />} />
          <Route path="/p/:name" element={<ViewerRoute />} />
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
