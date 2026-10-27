import { useCallback, useEffect, useState } from "react";

type State<T> = { data: T | null; error: unknown; loading: boolean };

// load must be stable (a module function or useCallback): it is the effect's dependency.
export function useLoad<T>(load: () => Promise<T>) {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let live = true;
    load().then(
      (data) => live && setState({ data, error: null, loading: false }),
      (error: unknown) => live && setState({ data: null, error, loading: false }),
    );
    return () => {
      live = false;
    };
  }, [load, tick]);

  const reload = useCallback(() => {
    setState((current) => ({ ...current, loading: true }));
    setTick((value) => value + 1);
  }, []);

  return { ...state, reload };
}

export function message(error: unknown): string | null {
  if (error == null) return null;
  return error instanceof Error ? error.message : String(error);
}
