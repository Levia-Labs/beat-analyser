import type { ReactNode } from "react";
import { AlertIcon } from "./icons";

// The shared look of the error page and the Not found page.
export default function StatusPage({
  title,
  detail,
  children,
}: {
  title: string;
  detail: string;
  children: ReactNode;
}) {
  return (
    <section className="card status-page" role="alert">
      <AlertIcon size={28} />
      <h1>{title}</h1>
      <p className="muted">{detail}</p>
      <div className="actions">{children}</div>
    </section>
  );
}
