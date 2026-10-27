import { useCallback, useEffect, useRef, useState } from "react";
import { getProjects, type ProjectItem } from "../api";
import { message } from "../useLoad";
import { apiParams, PAGE_SIZE, type ProjectsQuery } from "./query";

type Loaded = {
  version: string;
  items: ProjectItem[];
  total: number;
  error: string | null;
  more: boolean;
};

// A page belongs to the query and reload it was fetched for. Comparing versions, not resetting
// state inside the effect, drops rows of an older query the moment the query changes.
export function usePagedProjects(query: ProjectsQuery) {
  const [tick, setTick] = useState(0);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const fetching = useRef(false);
  const version = `${apiParams(query, 0, PAGE_SIZE).toString()}#${tick}`;

  useEffect(() => {
    let live = true;
    fetching.current = true;
    getProjects(apiParams(query, 0, PAGE_SIZE)).then(
      (page) => {
        if (!live) return;
        fetching.current = false;
        setLoaded({ version, items: page.items, total: page.total, error: null, more: false });
      },
      (error: unknown) => {
        if (!live) return;
        fetching.current = false;
        setLoaded({ version, items: [], total: 0, error: message(error), more: false });
      },
    );
    return () => {
      live = false;
    };
  }, [query, version]);

  const current = loaded?.version === version ? loaded : null;
  const items = current?.items ?? [];
  const total = current?.total ?? 0;
  const hasMore = current !== null && current.error === null && items.length < total;

  const loadMore = useCallback(() => {
    if (!current || fetching.current || current.items.length >= current.total) return;
    fetching.current = true;
    setLoaded({ ...current, more: true });
    getProjects(apiParams(query, current.items.length, PAGE_SIZE)).then(
      (page) => {
        fetching.current = false;
        setLoaded((now) =>
          now?.version === current.version
            ? { ...now, items: [...now.items, ...page.items], total: page.total, more: false }
            : now,
        );
      },
      (error: unknown) => {
        fetching.current = false;
        setLoaded((now) =>
          now?.version === current.version ? { ...now, error: message(error), more: false } : now,
        );
      },
    );
  }, [current, query]);

  const reload = useCallback(() => setTick((value) => value + 1), []);

  return {
    items,
    total,
    error: current?.error ?? null,
    loading: current === null,
    loadingMore: current?.more ?? false,
    hasMore,
    loadMore,
    reload,
  };
}
