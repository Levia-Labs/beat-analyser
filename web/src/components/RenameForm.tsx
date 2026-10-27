import { useState, type FormEvent } from "react";
import { CheckIcon, XIcon } from "./icons";

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
      <button type="submit" className="primary" disabled={busy || !value.trim()}>
        <CheckIcon /> Save
      </button>
      <button type="button" className="chip" onClick={onCancel} disabled={busy}>
        <XIcon /> Cancel
      </button>
    </form>
  );
}
