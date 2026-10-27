import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router";
import { parseQuery, toggleSort, toSearchParams, type ProjectsQuery, type SortKey } from "./query";

// Every writer reads the URL it is replacing (functional form), so a debounced write that fires
// late cannot undo a sort click made in between.
export function useProjectsQuery() {
  const [params, setParams] = useSearchParams();
  const query = useMemo(() => parseQuery(params), [params]);

  const update = useCallback(
    (patch: Partial<ProjectsQuery>, replace = false) =>
      setParams((current) => toSearchParams({ ...parseQuery(current), ...patch }), { replace }),
    [setParams],
  );
  const sortBy = useCallback(
    (column: SortKey) =>
      setParams((current) => toSearchParams(toggleSort(parseQuery(current), column))),
    [setParams],
  );
  // Clear resets the filters; the sort is a view choice and stays.
  const clear = useCallback(
    () =>
      setParams((current) => {
        const { sort, order } = parseQuery(current);
        return toSearchParams({ ...parseQuery(new URLSearchParams()), sort, order });
      }),
    [setParams],
  );

  return { query, update, sortBy, clear };
}
