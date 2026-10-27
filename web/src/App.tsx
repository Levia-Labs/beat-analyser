import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router";
import { getConfig, login, logout, onUnauthorized } from "./api";
import ErrorBoundary from "./components/ErrorBoundary";
import { LogoutIcon } from "./components/icons";
import Logo from "./components/Logo";
import SignIn from "./components/SignIn";
import Home from "./screens/Home";
import NotFound from "./screens/NotFound";
import { ViewerRoute } from "./screens/Viewer";

export default function App() {
  const [auth, setAuth] = useState(false);
  const [locked, setLocked] = useState(false);
  // Bumped on sign-in: remounting the screens makes each repeat the call that got the 401.
  const [session, setSession] = useState(0);

  useEffect(() => {
    let live = true;
    getConfig()
      .then((config) => live && setAuth(config.auth === true))
      .catch(() => {});
    const off = onUnauthorized(() => setLocked(true));
    return () => {
      live = false;
      off();
    };
  }, []);

  async function signOut() {
    try {
      await logout();
    } finally {
      setLocked(true);
    }
  }

  return (
    <>
      <header className="top">
        <Link to="/" className="brand">
          <Logo />
          <span>Beat Analyser</span>
        </Link>
        {auth && (
          <button type="button" className="chip sign-out" onClick={signOut}>
            <LogoutIcon /> Sign out
          </button>
        )}
      </header>
      <main className="page">
        {locked ? (
          <SignIn
            submit={login}
            onDone={() => {
              setAuth(true);
              setLocked(false);
              setSession((n) => n + 1);
            }}
          />
        ) : (
          <ErrorBoundary key={session}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/p/:name" element={<ViewerRoute />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </ErrorBoundary>
        )}
      </main>
    </>
  );
}
