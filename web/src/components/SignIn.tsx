import { useState, type FormEvent } from "react";
import Banner from "./Banner";
import { Icon } from "./icons";

export default function SignIn({
  submit,
  onDone,
}: {
  submit: (token: string) => Promise<void>;
  onDone: () => void;
}) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await submit(token);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sign in failed");
      setBusy(false);
      return;
    }
    onDone();
  }

  return (
    <form className="card sign-in" onSubmit={send}>
      <Icon name="lock" className="ic-lg" />
      <h1>Sign in</h1>
      <p className="muted">This server asks for an access token.</p>
      {error && <Banner message={error} />}
      <input
        className="field"
        type="password"
        aria-label="Access token"
        autoComplete="current-password"
        value={token}
        disabled={busy}
        onChange={(event) => setToken(event.target.value)}
      />
      <button type="submit" className="btn primary" disabled={busy || !token}>
        <Icon name="lock" /> Sign in
      </button>
    </form>
  );
}
