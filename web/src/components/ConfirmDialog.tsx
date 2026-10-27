import { useEffect, useRef } from "react";

// A plain element, not <dialog>: showModal is missing in jsdom, and this is one button pair.
export default function ConfirmDialog({
  title,
  body,
  confirm,
  busy,
  onConfirm,
  onCancel,
}: {
  title: string;
  body: string;
  confirm: string;
  busy: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => cancel.current?.focus(), []);
  // On the document, not the dialog: a keydown handler on a non-interactive element fails jsx-a11y.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);
  return (
    <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
      <p id="confirm-title">
        <strong>{title}</strong>
      </p>
      <p className="muted">{body}</p>
      <div className="actions">
        <button ref={cancel} type="button" className="chip" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
        <button type="button" className="primary" onClick={onConfirm} disabled={busy}>
          {confirm}
        </button>
      </div>
    </div>
  );
}
