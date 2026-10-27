import * as Checkbox from "@radix-ui/react-checkbox";
import { useId, useState } from "react";
import { CheckIcon } from "../components/icons";
import { ClearFilterButton } from "./FilterPopover";

// The Name field's text lives above the popover (see useDraftText), so closing the panel inside
// the debounce window does not lose it.
export type Draft = { value: string; set: (text: string) => void; clear: () => void };

export function TextFilterBody({ draft, active }: { draft: Draft; active: boolean }) {
  return (
    <div className="filter-body">
      <div className="filter-row">
        <input
          type="search"
          className="field"
          aria-label="Search by name"
          placeholder="Search by name"
          autoComplete="off"
          value={draft.value}
          onChange={(event) => draft.set(event.target.value)}
        />
        <ClearFilterButton column="Name" disabled={!active} onClear={draft.clear} />
      </div>
    </div>
  );
}

const NUMBER = /^\d+(\.\d+)?$/;
const filled = (text: string) => text.trim() !== "";
const badNumber = (text: string) => filled(text) && !NUMBER.test(text.trim());
const inverted = (min: string, max: string) =>
  filled(min) && filled(max) && !badNumber(min) && !badNumber(max) && Number(min) > Number(max);

function problem(min: string, max: string): string | null {
  if (badNumber(min) || badNumber(max)) return "Enter a number of 0 or more.";
  return inverted(min, max) ? "Minimum is above the maximum." : null;
}

// Only a valid pair reaches the URL: the field keeps what is typed, the query keeps the last good value.
export function RangeFilterBody({
  name,
  unit = "",
  min,
  max,
  onChange,
  onClear,
}: {
  name: string;
  unit?: string;
  min: string;
  max: string;
  onChange: (range: { min: string; max: string }, replace: boolean) => void;
  onClear: () => void;
}) {
  const [draft, setDraft] = useState({ min, max });
  const [seen, setSeen] = useState({ min, max });
  if (seen.min !== min || seen.max !== max) {
    // A change from outside (Clear, back button) replaces whatever is typed.
    setSeen({ min, max });
    if (draft.min.trim() !== min || draft.max.trim() !== max) setDraft({ min, max });
  }
  const error = problem(draft.min, draft.max);

  function edit(next: { min: string; max: string }) {
    setDraft(next);
    if (!problem(next.min, next.max)) {
      onChange({ min: next.min.trim(), max: next.max.trim() }, true);
    }
  }
  const field = (side: "min" | "max") => (
    <input
      type="text"
      className="field"
      inputMode="decimal"
      autoComplete="off"
      aria-label={`${name} ${side}${unit}`}
      placeholder={side === "min" ? "Min" : "Max"}
      aria-invalid={badNumber(draft[side]) || inverted(draft.min, draft.max)}
      value={draft[side]}
      onChange={(event) => edit({ ...draft, [side]: event.target.value })}
    />
  );
  return (
    <div className="filter-body">
      <div className="filter-row">
        {field("min")}
        {field("max")}
        <ClearFilterButton
          column={name}
          disabled={!min && !max && !filled(draft.min) && !filled(draft.max)}
          onClear={() => {
            setDraft({ min: "", max: "" });
            onClear();
          }}
        />
      </div>
      {error && (
        <p className="err" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export type Option = { value: string; label: string };

function Box({
  id,
  checked,
  onToggle,
  label,
}: {
  id: string;
  checked: boolean;
  onToggle: () => void;
  label: string;
}) {
  // The check icon is always mounted (forceMount) so the control is never an icon-less button.
  return (
    <div className="check">
      <Checkbox.Root id={id} className="checkbox" checked={checked} onCheckedChange={onToggle}>
        <Checkbox.Indicator forceMount className="checkbox-mark">
          <CheckIcon size={14} />
        </Checkbox.Indicator>
      </Checkbox.Root>
      <label htmlFor={id}>{label}</label>
    </div>
  );
}

export function ChecklistBody({
  group,
  options,
  selected,
  onChange,
  onClear,
}: {
  group: string;
  options: Option[];
  selected: string[];
  onChange: (next: string[]) => void;
  onClear: () => void;
}) {
  const base = useId();
  const toggle = (value: string) =>
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value]);
  return (
    <div className="filter-body">
      <div role="group" aria-label={group} className="checks">
        {options.map((option, index) => (
          <Box
            key={option.value}
            id={`${base}-${index}`}
            checked={selected.includes(option.value)}
            onToggle={() => toggle(option.value)}
            label={option.label}
          />
        ))}
      </div>
      <div className="filter-row end">
        <ClearFilterButton column={group} disabled={selected.length === 0} onClear={onClear} />
      </div>
    </div>
  );
}
