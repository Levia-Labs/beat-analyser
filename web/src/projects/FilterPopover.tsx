import * as Popover from "@radix-ui/react-popover";
import type { ReactNode } from "react";
import { Icon } from "../components/icons";

// Icon-only clear button placed beside a filter's fields; one per column.
export function ClearFilterButton({
  column,
  disabled,
  onClear,
}: {
  column: string;
  disabled: boolean;
  onClear: () => void;
}) {
  return (
    <button
      type="button"
      className="btn icon clear-filter"
      aria-label={`Clear ${column} filter`}
      title="Clear this filter"
      disabled={disabled}
      onClick={onClear}
    >
      <Icon name="filter-x" />
    </button>
  );
}

// A column header's filter button: opens an anchored popover whose first field takes focus.
export default function FilterPopover({
  label,
  count,
  children,
}: {
  label: string;
  count: number;
  children: ReactNode;
}) {
  const active = count > 0;
  return (
    <Popover.Root>
      <Popover.Trigger
        className={active ? "filter-btn active" : "filter-btn"}
        aria-label={active ? `Filter ${label}, ${count} active` : `Filter ${label}`}
      >
        <Icon name="filter" />
        {active && (
          <span className="filter-badge" aria-hidden="true">
            {count}
          </span>
        )}
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="popover"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          aria-label={`${label} filter`}
          onOpenAutoFocus={(event) => {
            // The first field, not the close button: Escape then closes the panel at once.
            const panel = event.currentTarget as HTMLElement;
            const field = panel.querySelector<HTMLElement>(
              "input, [role='checkbox'], button:not(.popover-x):not(.clear-filter)",
            );
            event.preventDefault();
            (field ?? panel).focus();
          }}
        >
          <header className="popover-head">
            <h3 className="popover-title">{label}</h3>
            <Popover.Close className="btn plain icon popover-x" aria-label="Close filter">
              <Icon name="x" />
            </Popover.Close>
          </header>
          {children}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
