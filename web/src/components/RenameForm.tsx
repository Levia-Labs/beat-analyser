import { useState, type FormEvent } from "react";
import { Icon } from "./icons";

export default function RenameForm({
  name,
  busy,
  onSave,
  onCancel,
}: {
  name: string;
  busy: boolean;
  onSave: (to: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(name);
  function submit(event: FormEvent) {
    event.preventDefault();
    if (value.trim()) onSave(value.trim());
  }
  return (
    <form className="actions" onSubmit={submit}>
      <input
        className="field"
        aria-label="New name"
        value={value}
        maxLength={100}
        disabled={busy}
        onChange={(event) => setValue(event.target.value)}
      />
      <button type="submit" className="btn primary" disabled={busy || !value.trim()}>
        <Icon name="check" /> Save
      </button>
      <button type="button" className="btn" onClick={onCancel} disabled={busy}>
        <Icon name="x" /> Cancel
      </button>
    </form>
  );
}
