import { useEffect, useState } from "react";

export const DEBOUNCE_MS = 250;

// Text being typed into a URL-backed filter. The URL is the truth: the field follows it when it
// changed from outside (Clear, back button) and writes it after a pause, replacing the history entry
// so typing does not fill the back button.
export function useDraftText(value: string, commit: (text: string, replace: boolean) => void) {
  const [draft, setDraft] = useState(value);
  const [seen, setSeen] = useState(value);
  if (seen !== value) {
    setSeen(value);
    setDraft(value);
  }

  useEffect(() => {
    if (draft === value) return;
    const timer = setTimeout(() => commit(draft, true), DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [draft, value, commit]);

  return {
    value: draft,
    set: setDraft,
    // Empties the field without writing the URL: the caller is about to clear it another way.
    reset: () => setDraft(""),
    // Resetting the draft re-runs the effect, which cancels a pending write of the typed text; the
    // URL change alone would miss a URL that was already empty.
    clear: () => {
      setDraft("");
      commit("", false);
    },
  };
}
