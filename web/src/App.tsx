import { Link, Route, Routes } from "react-router";
import ErrorBoundary from "./components/ErrorBoundary";
import Logo from "./components/Logo";
import NavTabs from "./components/NavTabs";
import Add from "./screens/Add";
import Home from "./screens/Home";
import NotFound from "./screens/NotFound";
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
      <NavTabs />
      <main className="page">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/add" element={<Add />} />
            <Route path="/p/:name" element={<ViewerRoute />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </>
  );
}
