import { Component, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router";
import { Icon } from "./icons";
import StatusPage from "./StatusPage";

function ErrorPage({ error, clear }: { error: Error; clear: () => void }) {
  const navigate = useNavigate();
  return (
    <StatusPage title="Something went wrong" detail={error.message || "Unexpected error"}>
      <button type="button" className="btn" onClick={() => window.location.reload()}>
        <Icon name="refresh" /> Reload
      </button>
      <button
        type="button"
        className="btn"
        onClick={() => {
          clear();
          navigate("/");
        }}
      >
        <Icon name="list" /> Projects
      </button>
    </StatusPage>
  );
}

type Props = { resetKey: string; children: ReactNode };

class Boundary extends Component<Props, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  // A route change drops the error so the next page gets a fresh try.
  override componentDidUpdate(previous: Props) {
    if (this.state.error && previous.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  override render() {
    const { error } = this.state;
    if (error) return <ErrorPage error={error} clear={() => this.setState({ error: null })} />;
    return this.props.children;
  }
}

export default function ErrorBoundary({ children }: { children: ReactNode }) {
  return <Boundary resetKey={useLocation().key}>{children}</Boundary>;
}
