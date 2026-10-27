import { Icon } from "./icons";

// Icon only: the tone bar and the glyph carry the state, the label is for assistive technology.
export function StatusChip({ done }: { done: boolean }) {
  const label = done ? "Built" : "Not built";
  return (
    <span className={done ? "chip chip-ok" : "chip chip-error"} role="img" aria-label={label}>
      <Icon name={done ? "ok" : "close-circle"} />
    </span>
  );
}
