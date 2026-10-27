import { useCallback, useEffect, useRef, useState } from "react";
import { getProjects, type ProjectItem } from "../api";
import { message } from "../useLoad";
import { apiParams, PAGE_SIZE, type ProjectsQuery } from "./query";

// The most rows one in-place reload asks for: the API's page ceiling.
const RELOAD_CAP = 200;

type Loaded = {
  version: string;
  items: ProjectItem[];
  total: number;
  error: string | null;
  more: boolean;
};

// Offset paging over a list that changes: a build, rename or delete between pages shifts rows,
// so a page can repeat rows already shown. Keep the first of each name.
function append(items: ProjectItem[], page: ProjectItem[]): ProjectItem[] {
  const seen = new Set(items.map((item) => item.name));
  const fresh = page.filter((item) => !seen.has(item.name));
  return fresh.length === 0 ? items : [...items, ...fresh];
}

// A page belongs to the query it was fetched for. Comparing versions, not resetting state inside
// the effect, drops rows of an older query the moment the query changes.
export function usePagedProjects(query: ProjectsQuery) {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const fetching = useRef(false);
  // Bumped by a reload: a page request started before it must not write over fresher rows.
  const epoch = useRef(0);
  const latest = useRef<Loaded | null>(null);
  const version = apiParams(query, 0, PAGE_SIZE).toString();

  useEffect(() => {
    latest.current = loaded;
  });

  useEffect(() => {
    let live = true;
    epoch.current += 1;
    fetching.current = true;
    getProjects(apiParams(query, 0, PAGE_SIZE)).then(
      (page) => {
        if (!live) return;
        fetching.current = false;
        setLoaded({
          version,
          items: append([], page.items),
          total: page.total,
          error: null,
          more: false,
        });
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

  const fetchPages = useCallback(
    async (start: Loaded) => {
      const mine = epoch.current;
      const stale = () => epoch.current !== mine;
      fetching.current = true;
      setLoaded({ ...start, error: null, more: true });
      let rows = start.items;
      let count = start.total;
      try {
        for (;;) {
          const page = await getProjects(apiParams(query, rows.length, PAGE_SIZE));
          if (stale()) return;
          const next = append(rows, page.items);
          const added = next.length - rows.length;
          rows = next;
          count = page.total;
          setLoaded((now) =>
            now?.version === start.version ? { ...now, items: rows, total: count } : now,
          );
          // A shifted list can leave the page short after the dedupe: ask again from the new end.
          if (added === page.items.length || page.items.length === 0 || rows.length >= count) break;
        }
        fetching.current = false;
        setLoaded((now) => (now?.version === start.version ? { ...now, more: false } : now));
      } catch (error) {
        if (stale()) return;
        fetching.current = false;
        setLoaded((now) =>
          now?.version === start.version ? { ...now, error: message(error), more: false } : now,
        );
      }
    },
    [query],
  );

  const loadMore = useCallback(() => {
    if (!current || fetching.current || current.items.length >= current.total) return;
    void fetchPages(current);
  }, [current, fetchPages]);

  // Retry the page that failed: the rows already shown stay, the same offset is asked again.
  const retry = useCallback(() => {
    const now = latest.current;
    if (!now || now.version !== version) return;
    void fetchPages(now);
  }, [fetchPages, version]);

  // In place: refetch the range already loaded and swap it in when it arrives, so the rows and
  // the scroll position stay put meanwhile.
  const reload = useCallback(() => {
    const now = latest.current;
    if (!now || now.version !== version) return;
    epoch.current += 1;
    const mine = epoch.current;
    fetching.current = false;
    const limit = Math.min(RELOAD_CAP, Math.max(PAGE_SIZE, now.items.length));
    getProjects(apiParams(query, 0, limit)).then(
      (page) => {
        if (epoch.current !== mine) return;
        setLoaded((was) =>
          was?.version === version
            ? { ...was, items: append([], page.items), total: page.total, error: null, more: false }
            : was,
        );
      },
      (error: unknown) => {
        if (epoch.current !== mine) return;
        setLoaded((was) => (was?.version === version ? { ...was, error: message(error) } : was));
      },
    );
  }, [query, version]);

  return {
    items,
    total,
    error: current?.error ?? null,
    loading: current === null,
    loadingMore: current?.more ?? false,
    hasMore,
    loadMore,
    retry,
    reload,
  };
}
