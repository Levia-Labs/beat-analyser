import { useEffect } from "react";

// Sets the tab title while the caller is mounted and puts the previous one back on leaving.
export function useDocumentTitle(title: string) {
  useEffect(() => {
    const previous = document.title;
    document.title = title;
    return () => {
      document.title = previous;
    };
  }, [title]);
}
