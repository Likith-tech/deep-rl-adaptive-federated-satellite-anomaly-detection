import { useEffect, useState } from "react";
import { ApiError } from "@/api/client";

export type ResourceState = "loading" | "ready" | "error";

interface UseApiResourceResult<T> {
  state: ResourceState;
  data: T | null;
  error: string | null;
}

/**
 * Generic real-data fetch hook used by every OrbitShield page backed by
 * a live API resource (as opposed to PlaceholderPage, used where no
 * backing system exists yet). `deps` re-runs the fetch when it changes
 * (e.g. a route param), mirroring React Query's key semantics without
 * adding a dependency for what this project's read-only, unauthenticated
 * GET endpoints need.
 */
export function useApiResource<T>(fetcher: () => Promise<T>, deps: unknown[] = []): UseApiResourceResult<T> {
  const [state, setState] = useState<ResourceState>("loading");
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setState("loading");

    fetcher()
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setState("ready");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Failed to load data");
        setState("error");
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { state, data, error };
}
